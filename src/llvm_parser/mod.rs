use crate::{
    llvm_parser::{
        block_mapper::{BasicBlockId, BasicBlockMapper},
        control_flow::scopes::SymbolScopes,
        instructions::RegionLowerer,
        scc::SccTree,
    },
    rvsdg::{
        GlobalId, GlobalInit, InlineHint, Linkage, RVSDGMod, ThreadLocalMode, Visibility,
        func::{
            CallingConvention, FnAttrFlags, FnAttrs, FnDecl, MemoryEffects, ModRef, Param,
            ParamAttrFlags, ParamAttrs, ParamAttrsExtra,
        },
        global::{DllStorageClass, GlobalDef, UnnamedAddr},
        types::{
            ArrayType, FuncType, PtrType, ScalarType, StructDef, StructField, TypeArena, TypeRef,
            VOID, VectorType,
        },
    },
};
use color_eyre::eyre::eyre;
use llvm_ir::{Module, TypeRef as LLVMTypeRef, function::FunctionDeclaration};
use std::str::FromStr;
use target_lexicon::Triple;

pub mod block_mapper;
pub mod call_instructions;
pub mod const_instructions;
pub mod control_flow;
pub mod instructions;
pub mod scc;
#[cfg(test)]
pub mod test_utils;
pub mod vector_instructions;

struct FnCtx<'m> {
    pub llvm_mod: &'m Module,
    pub func: &'m llvm_ir::Function,
    pub bb_mapper: &'m BasicBlockMapper,
    /// Tree of every non-trivial strongly connected component in the
    /// function, with parent / child nesting. The source of truth for loop
    /// detection. (The synthetic exit block id lives on the overlay, which
    /// every consumer of it also holds.)
    pub scc_tree: &'m SccTree,
}

/// Intern every basic block of `func` to a dense [`BasicBlockId`], append the
/// synthetic exit block, and add the CFG arcs of every block **reachable from
/// the entry** -- including each `Ret`'s arc to that synthetic exit. This
/// establishes the *closed CFG* (Bahmann, Reissmann, Jahre, Meyer 2015, Def 2.1:
/// a unique entry with no predecessors and a unique exit with no successors) that
/// the restructuring transform and construction walk operate on. Block 0 is the
/// unique entry; the synthetic exit (`BasicBlockMapper::exit_name`) is the unique
/// exit, recoverable from the returned mapper via `get_exit_expect`.
///
/// Arcs out of **unreachable** blocks are deliberately omitted. `clang
/// -disable-llvm-passes` leaves dead blocks in place (no CFG cleanup runs), and
/// an arc from a dead block *into* a loop fabricates a second entry vertex that
/// makes a reducible loop look irreducible. Dead code cannot affect behaviour, so
/// dropping its arcs is sound and keeps the loop analysis honest. Unreachable
/// blocks stay interned (ids must keep matching `func.basic_blocks` indices) but
/// end up arc-isolated, so the construction walk never reaches them.
#[tracing::instrument(name = "intern_blocks", skip_all, fields(blocks = func.basic_blocks.len()))]
fn intern_blocks_and_arcs(func: &llvm_ir::Function) -> BasicBlockMapper {
    let mut bb_mapper = BasicBlockMapper::new(func.basic_blocks.len());
    for block in &func.basic_blocks {
        bb_mapper.intern(&block.name);
    }

    // append the fake exit block to the blocks.
    // this is used to map return values
    let exit_block_id = {
        let fake_exit_name = bb_mapper.exit_name();
        debug_assert!(
            bb_mapper.get(&fake_exit_name).is_none(),
            "basic block used reserved name {fake_exit_name}"
        );
        bb_mapper.intern(&fake_exit_name)
    };

    // Mark the blocks reachable from the entry (block 0) by walking terminator
    // successors. Arcs are added only for these, so dead blocks stay isolated.
    let mut reachable = vec![false; func.basic_blocks.len()];
    reachable[0] = true;
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        let mut succ_names: Vec<&llvm_ir::Name> = Vec::new();
        match &func.basic_blocks[i].term {
            llvm_ir::Terminator::Br(br) => succ_names.push(&br.dest),
            llvm_ir::Terminator::CondBr(cond_br) => {
                succ_names.push(&cond_br.true_dest);
                succ_names.push(&cond_br.false_dest);
            }
            llvm_ir::Terminator::Switch(switch) => {
                succ_names.push(&switch.default_dest);
                for (_, dest) in &switch.dests {
                    succ_names.push(dest);
                }
            }
            _ => {}
        }
        for name in succ_names {
            let succ = bb_mapper.get_expect(name).0 as usize;
            if !reachable[succ] {
                reachable[succ] = true;
                stack.push(succ);
            }
        }
    }

    for (i, block) in func.basic_blocks.iter().enumerate() {
        if !reachable[i] {
            continue;
        }
        let from = BasicBlockId(i as u32);
        match &block.term {
            llvm_ir::Terminator::Br(br) => {
                let to = *bb_mapper.get_expect(&br.dest);
                bb_mapper.add_connection(from, to);
            }
            llvm_ir::Terminator::CondBr(cond_br) => {
                let true_block = *bb_mapper.get_expect(&cond_br.true_dest);
                let false_block = *bb_mapper.get_expect(&cond_br.false_dest);
                bb_mapper.add_connection(from, true_block);
                bb_mapper.add_connection(from, false_block);
            }
            llvm_ir::Terminator::Ret(_) => {
                bb_mapper.add_connection(from, exit_block_id);
            }
            llvm_ir::Terminator::Switch(switch) => {
                let default = *bb_mapper.get_expect(&switch.default_dest);
                bb_mapper.add_connection(from, default);
                for (_, dest) in &switch.dests {
                    let dest_id = *bb_mapper.get_expect(dest);
                    bb_mapper.add_connection(from, dest_id);
                }
            }
            llvm_ir::Terminator::Unreachable(_) => (),
            llvm_ir::Terminator::Invoke(_) => todo!(),
            other => todo!("handle terminator case: {other:?}"),
        }
    }

    bb_mapper
}

impl RVSDGMod {
    #[tracing::instrument(skip_all)]
    pub fn from_llvm_mod(module: Module) -> color_eyre::Result<RVSDGMod> {
        let mut rvsdg_mod = match &module.target_triple {
            Some(triple) => RVSDGMod::new(
                module.name.clone(),
                Triple::from_str(triple).map_err(|e| {
                    eyre!("Failed to convert llvm triple into target_lexicon triple: {e}")
                })?,
                module.data_layout.layout_str.clone(),
            ),
            None => RVSDGMod::new_host(module.name.clone()),
        };
        // Module-level inline assembly defines real symbols (hand-written
        // routines the module calls); preserve it verbatim.
        rvsdg_mod.module_asm = module.inline_assembly.clone();

        // lower function declerations
        for func in &module.func_declarations {
            let decl = FnDecl::from_declaration(func, &mut rvsdg_mod.tables.types, &module)?;
            rvsdg_mod.declare_fn_full(decl);
        }
        for func in &module.functions {
            let decl = FnDecl::from_fn(func, &mut rvsdg_mod.tables.types, &module)?;
            rvsdg_mod.declare_fn_full(decl);
        }

        // Globals are lowered in two passes so an initializer can reference
        // a global declared later in the module (csmith emits such forward
        // references freely -- e.g. one global initialised with the
        // address of another). Pass 1 registers every global's name (and
        // its type) in `global_map`; pass 2 resolves the initializers,
        // which may now look up any global by name. (Aliases and ifuncs
        // are refused below, so no other symbol kinds exist here.)

        // Pass 1: register names with a placeholder (Extern) initializer.
        let mut var_ids: Vec<GlobalId> = Vec::with_capacity(module.global_vars.len());
        for global in &module.global_vars {
            // Exhaustive destructure: adding a field to llvm-ir's
            // GlobalVariable breaks compilation HERE until it is carried or
            // deliberately dropped. This is the completeness contract that
            // stops attributes silently vanishing one broken build at a
            // time (thread_local and module asm both got in that way).
            let llvm_ir::module::GlobalVariable {
                name,
                linkage,
                visibility,
                is_constant,
                // The global value's own type; superseded by `value_type`
                // below (LLVMGlobalGetValueType), which is the authoritative
                // content type and needs no derivation from the initializer.
                ty: _ty,
                addr_space,
                dll_storage_class,
                thread_local_mode,
                unnamed_addr,
                initializer,
                section,
                // COMDAT selection only matters for C++ ODR-style link
                // dedup; linkonce/weak LINKAGE (carried above) covers the C
                // corpus. Dropped until a real input needs it.
                comdat: _comdat,
                alignment,
                // No debug info support yet.
                debugloc: _debugloc,
                value_type,
            } = global;

            let value_ty = match initializer {
                // LLVM requires a global's type and its initializer's type
                // to match EXACTLY, and the initializer is lowered from its
                // own type (pass 2), so that type is authoritative here --
                // value_type can legitimately differ in named-ness (a named
                // struct global initialised by a literal struct constant).
                Some(init) => {
                    let ty = module.types.type_of(init.as_ref());
                    rvsdg_mod.tables.types.convert_type_ref(&ty, &module)?
                }
                // No initializer (external/declared global): value_type
                // (LLVMGlobalGetValueType) is the authoritative content
                // type; the global value's own type is just an opaque ptr.
                None => rvsdg_mod
                    .tables
                    .types
                    .convert_type_ref(value_type, &module)?,
            };
            let id = rvsdg_mod.tables.define_global(GlobalDef {
                name: global_name_string(name),
                ty: value_ty,
                // Resolved in pass 2, once every name is registered.
                initializer: GlobalInit::Extern,
                is_constant: *is_constant,
                linkage: convert_linkage(*linkage),
                alignment: (*alignment != 0).then_some(*alignment),
                section: section.clone(),
                visibility: convert_visibility(*visibility),
                thread_local: convert_thread_local_mode(*thread_local_mode),
                addr_space: *addr_space,
                unnamed_addr: match unnamed_addr {
                    None => UnnamedAddr::None,
                    Some(llvm_ir::module::UnnamedAddr::Local) => UnnamedAddr::Local,
                    Some(llvm_ir::module::UnnamedAddr::Global) => UnnamedAddr::Global,
                },
                dll_storage_class: convert_dll_storage_class(*dll_storage_class),
            });
            var_ids.push(id);
        }
        // Aliases and ifuncs are refused, not mistranslated. An alias is a
        // second NAME for an existing definition (a linker-level construct,
        // not a computational one); an earlier encoding here (a global
        // variable holding the aliasee's ADDRESS) silently changed
        // semantics for pointer-typed aliases: loads through the alias read
        // an address instead of the data, and the two symbols stopped
        // being identical. An ifunc picks its implementation by running a
        // resolver at LOAD time, so it cannot be resolved during
        // compilation at all.
        //
        // The settled design, when an input needs it: aliases never enter
        // the graph. In-module references to a NON-interposable alias
        // (private/internal) resolve directly to the aliasee -- exact,
        // since the two share one address, and better for the optimiser
        // than an opaque second symbol. In-module references to an
        // interposable (weak/external) alias stay refused: folding them
        // would bypass link-time interposition. Every alias is carried as
        // passive module metadata (like module_asm) and re-emitted for the
        // symbol table via LLVMAddAlias2, which inkwell does not wrap (one
        // contained llvm-sys call). Body copies or wrapper functions are
        // NOT valid substitutes: both break the address identity
        // (&alias == &aliasee) that the construct guarantees.
        if let Some(alias) = module.global_aliases.first() {
            return Err(eyre!(
                "module defines the alias {:?}; aliases are not supported yet",
                alias.name
            ));
        }
        if let Some(ifunc) = module.global_ifuncs.first() {
            return Err(eyre!(
                "module defines the ifunc {:?}; ifuncs are not supported yet",
                ifunc.name
            ));
        }

        // Pass 2: resolve initializers now that every name is registered.
        for (global, &id) in module.global_vars.iter().zip(&var_ids) {
            if let Some(init) = &global.initializer {
                let cid = rvsdg_mod.tables.convert_const_ref(init.clone(), &module)?;
                rvsdg_mod.tables.set_global_init(id, GlobalInit::Init(cid));
            }
        }
        // TODO: lower types
        // pub types: Types,

        // lower function bodies
        for func in &module.functions {
            rvsdg_mod.lower_fn_body(func, &module)?;
        }

        Ok(rvsdg_mod)
    }

    #[tracing::instrument(skip_all, fields(func = %func.name))]
    fn lower_fn_body(
        &mut self,
        func: &llvm_ir::Function,
        module: &Module,
    ) -> color_eyre::Result<()> {
        let fn_id = self
            .tables
            .get_function_by_name(&func.name)
            .ok_or_else(|| color_eyre::eyre::eyre!("function `{}` was not declared", func.name))?
            .id;
        let bb_mapper = intern_blocks_and_arcs(func);

        // Build the strongly connected component tree. This performs the
        // whole-function Tarjan pass plus one sub-Tarjan per non-trivial
        // component to recover nested-loop structure. See the `scc` module
        // for the algorithm.
        let scc_tree = SccTree::build(&bb_mapper);

        let fn_ctx = FnCtx {
            llvm_mod: module,
            func,
            bb_mapper: &bb_mapper,
            scc_tree: &scc_tree,
        };

        // Two-phase construction: restructure the control flow into overlay
        // records (loop pass then branch pass), then emit the RVSDG by
        // walking the restructured graph.
        let diverging: Vec<bool> = func
            .basic_blocks
            .iter()
            .map(|block| matches!(block.term, llvm_ir::Terminator::Unreachable(_)))
            // The synthetic exit block never diverges.
            .chain(std::iter::once(false))
            .collect();
        let overlay = control_flow::build_overlay(&bb_mapper, &scc_tree, diverging);
        self.define_fn(fn_id, |rb| {
            let mut scopes = SymbolScopes::new(rb.region_id);
            // Register function parameters as root-frame bindings.
            for (i, param) in func.parameters.iter().enumerate() {
                let value = rb.param(i as u32);
                scopes.bind_name(&param.name, value);
            }
            let mut builder = RegionLowerer::new(rb, &mut scopes, &fn_ctx);
            control_flow::emit::emit_function_body(&mut builder, &overlay)
        })
    }
}
impl TypeArena {
    fn convert_struct_fields(
        &mut self,
        element_types: &[LLVMTypeRef],
        module: &Module,
    ) -> color_eyre::Result<Vec<StructField>> {
        element_types
            .iter()
            .enumerate()
            .map(|(i, t)| {
                Ok(StructField {
                    name: None,
                    index: i as u64,
                    field_type: self.convert_type_ref(t, module)?,
                })
            })
            .collect()
    }

    fn convert_type_ref(
        &mut self,
        ty: &LLVMTypeRef,
        module: &Module,
    ) -> color_eyre::Result<TypeRef> {
        Ok(match ty.as_ref() {
            llvm_ir::Type::VoidType => VOID,
            llvm_ir::Type::IntegerType { bits } => TypeRef::Scalar(int_bit_to_scalar(*bits)?),
            llvm_ir::Type::PointerType { addr_space: _ } => {
                // Opaque pointers in LLVM 17+ -- no pointee type
                TypeRef::Ptr(self.intern_ptr(PtrType {
                    pointee: None,
                    alias_set: None,
                    no_escape: false,
                }))
            }
            llvm_ir::Type::FPType(fptype) => match fptype {
                llvm_ir::types::FPType::Single => TypeRef::Scalar(ScalarType::F32),
                llvm_ir::types::FPType::Double => TypeRef::Scalar(ScalarType::F64),
                llvm_ir::types::FPType::X86_FP80 => TypeRef::Scalar(ScalarType::F80),
                other => Err(eyre!("unsupported float type: {other:?}"))?,
            },
            llvm_ir::Type::FuncType {
                result_type,
                param_types,
                is_var_arg,
            } => {
                let ret = self.convert_type_ref(result_type, module)?;
                let params: Vec<TypeRef> = param_types
                    .iter()
                    .map(|t| self.convert_type_ref(t, module))
                    .collect::<color_eyre::Result<_>>()?;
                TypeRef::Func(self.intern_fn(FuncType {
                    params,
                    ret,
                    is_var_arg: *is_var_arg,
                }))
            }
            llvm_ir::Type::VectorType {
                element_type,
                num_elements,
                scalable,
            } => {
                if *scalable {
                    return Err(eyre!("scalable vectors not yet supported"));
                }
                let element = self.convert_type_ref(element_type, module)?;
                TypeRef::Vector(self.intern_vector(VectorType {
                    element,
                    lanes: *num_elements as u32,
                }))
            }
            llvm_ir::Type::ArrayType {
                element_type,
                num_elements,
            } => {
                let element = self.convert_type_ref(element_type, module)?;
                TypeRef::Array(self.intern_array(ArrayType {
                    element,
                    len: *num_elements as u64,
                }))
            }
            llvm_ir::Type::StructType {
                element_types,
                is_packed,
            } => {
                let fields = self.convert_struct_fields(element_types, module)?;
                TypeRef::Struct(self.intern_struct(StructDef {
                    name: None,
                    fields,
                    // calculating the struct size here requires knowing the offsets and
                    // padding which LLVM will do for us, so we only need this when not going into
                    // LLVM.
                    size: 0,
                    packed: *is_packed,
                }))
            }
            llvm_ir::Type::NamedStructType { name } => {
                match module.types.named_struct_def(name) {
                    // Keep the NAME on the interned struct: emission
                    // recreates it as an LLVM named struct, so the type
                    // round-trips as %name instead of decaying to an
                    // anonymous literal struct.
                    Some(llvm_ir::types::NamedStructDef::Defined(inner_ty)) => {
                        match inner_ty.as_ref() {
                            llvm_ir::Type::StructType {
                                element_types,
                                is_packed,
                            } => {
                                let fields = self.convert_struct_fields(element_types, module)?;
                                TypeRef::Struct(self.intern_struct(StructDef {
                                    name: Some(name.clone()),
                                    fields,
                                    size: 0,
                                    packed: *is_packed,
                                }))
                            }
                            other => {
                                return Err(eyre!(
                                    "named struct '{name}' defined as a non-struct type: {other:?}"
                                ));
                            }
                        }
                    }
                    Some(llvm_ir::types::NamedStructDef::Opaque) => {
                        // Opaque structs are only used behind pointers, so treat
                        // as an empty struct placeholder
                        TypeRef::Struct(self.intern_struct(StructDef {
                            name: Some(name.clone()),
                            fields: vec![],
                            size: 0,
                            packed: false,
                        }))
                    }
                    None => return Err(eyre!("named struct '{name}' not found in module")),
                }
            }
            // Target-specific and metadata types have no RVSDG representation
            llvm_ir::Type::X86_MMXType => return Err(eyre!("x86_mmx type not supported")),
            llvm_ir::Type::X86_AMXType => return Err(eyre!("x86_amx type not supported")),
            llvm_ir::Type::MetadataType => {
                return Err(eyre!("metadata type not supported in value context"));
            }
            llvm_ir::Type::LabelType => {
                return Err(eyre!("label type not supported in value context"));
            }
            llvm_ir::Type::TokenType => return Err(eyre!("token type not supported")),
            llvm_ir::Type::TargetExtType => {
                return Err(eyre!("target extension type not supported"));
            }
        })
    }
}

/// The function-signature fields shared by `llvm_ir::Function` and
/// `FunctionDeclaration`. Implementing it for both lets the signature ->
/// `FnDecl` conversion read the fields by reference from either, instead of
/// cloning a whole temporary `FunctionDeclaration` per defined function.
trait FnSignature {
    fn sig_name(&self) -> &str;
    fn sig_parameters(&self) -> &[llvm_ir::function::Parameter];
    fn sig_is_var_arg(&self) -> bool;
    fn sig_return_type(&self) -> &LLVMTypeRef;
    fn sig_return_attributes(&self) -> &[llvm_ir::function::ParameterAttribute];
    fn sig_linkage(&self) -> llvm_ir::module::Linkage;
    fn sig_visibility(&self) -> llvm_ir::module::Visibility;
    fn sig_calling_convention(&self) -> llvm_ir::function::CallingConvention;
    fn sig_alignment(&self) -> u32;
    fn sig_dll_storage_class(&self) -> llvm_ir::module::DLLStorageClass;
    fn sig_attributes(&self) -> &[llvm_ir::function::FunctionAttribute];
}

macro_rules! impl_fn_signature {
    ($t:ty) => {
        impl FnSignature for $t {
            fn sig_name(&self) -> &str {
                &self.name
            }
            fn sig_parameters(&self) -> &[llvm_ir::function::Parameter] {
                &self.parameters
            }
            fn sig_is_var_arg(&self) -> bool {
                self.is_var_arg
            }
            fn sig_return_type(&self) -> &LLVMTypeRef {
                &self.return_type
            }
            fn sig_return_attributes(&self) -> &[llvm_ir::function::ParameterAttribute] {
                &self.return_attributes
            }
            fn sig_linkage(&self) -> llvm_ir::module::Linkage {
                self.linkage
            }
            fn sig_visibility(&self) -> llvm_ir::module::Visibility {
                self.visibility
            }
            fn sig_calling_convention(&self) -> llvm_ir::function::CallingConvention {
                self.calling_convention
            }
            fn sig_alignment(&self) -> u32 {
                self.alignment
            }
            fn sig_dll_storage_class(&self) -> llvm_ir::module::DLLStorageClass {
                self.dll_storage_class
            }
            fn sig_attributes(&self) -> &[llvm_ir::function::FunctionAttribute] {
                &self.function_attributes
            }
        }
    };
}
impl_fn_signature!(FunctionDeclaration);
impl_fn_signature!(llvm_ir::Function);

/// Convert a parameter's (or return value's) LLVM attribute list into the
/// RVSDG representation. The ABI-bearing attributes matter most: `byval`
/// (the caller stack-copies the pointee), `sret` (hidden struct-return
/// slot), and `zeroext`/`signext` (sub-register integers are extended at
/// the call boundary). Dropping any of these miscompiles calls that cross
/// an externally-compiled boundary.
fn convert_param_attrs(
    attributes: &[llvm_ir::function::ParameterAttribute],
    types: &mut TypeArena,
    module: &Module,
) -> color_eyre::Result<ParamAttrs> {
    use llvm_ir::function::ParameterAttribute;
    let mut flags = ParamAttrFlags::empty();
    let mut by_value = None;
    let mut struct_return = None;
    let mut alignment = None;
    let mut dereferenceable_bytes = None;
    for attr in attributes {
        match attr {
            ParameterAttribute::ZeroExt => flags |= ParamAttrFlags::ZERO_EXTEND,
            ParameterAttribute::SignExt => flags |= ParamAttrFlags::SIGN_EXTEND,
            ParameterAttribute::NoAlias => flags |= ParamAttrFlags::NO_ALIAS,
            ParameterAttribute::NoCapture => flags |= ParamAttrFlags::NO_CAPTURE,
            ParameterAttribute::NonNull => flags |= ParamAttrFlags::NON_NULL,
            ParameterAttribute::NoUndef => flags |= ParamAttrFlags::NO_UNDEF,
            ParameterAttribute::Returned => flags |= ParamAttrFlags::RETURNED,
            ParameterAttribute::ByVal(ty) => {
                by_value = Some(types.convert_type_ref(ty, module)?);
            }
            ParameterAttribute::SRet(ty) => {
                struct_return = Some(types.convert_type_ref(ty, module)?);
            }
            ParameterAttribute::Alignment(bytes) => alignment = Some(*bytes as u32),
            ParameterAttribute::Dereferenceable(bytes) => {
                dereferenceable_bytes = Some(*bytes);
            }
            // Optimisation-only attributes we do not model yet; dropping
            // them is conservative (never changes the ABI).
            _ => {}
        }
    }
    let extra = (by_value.is_some()
        || struct_return.is_some()
        || alignment.is_some()
        || dereferenceable_bytes.is_some())
    .then(|| {
        Box::new(ParamAttrsExtra {
            by_value,
            struct_return,
            alignment,
            dereferenceable_bytes,
            range: None,
        })
    });
    Ok(ParamAttrs { flags, extra })
}

impl FnDecl {
    fn from_signature<S: FnSignature + ?Sized>(
        func: &S,
        types: &mut TypeArena,
        module: &Module,
    ) -> color_eyre::Result<Self> {
        let ret_ty = types.convert_type_ref(func.sig_return_type(), module)?;
        let return_types = if ret_ty == VOID { vec![] } else { vec![ret_ty] };

        let mut decl = Self {
            name: func.sig_name().to_string(),
            params: func
                .sig_parameters()
                .iter()
                .map(|param| {
                    let attrs = convert_param_attrs(&param.attributes, types, module)?;
                    Ok(Param {
                        ty: types.convert_type_ref(&param.ty, module)?,
                        flags: attrs.flags,
                        extra: attrs.extra,
                    })
                })
                .collect::<color_eyre::Result<_>>()?,
            return_types,
            return_attrs: convert_param_attrs(func.sig_return_attributes(), types, module)?,
            linkage_type: convert_linkage(func.sig_linkage()),
            calling_convention: convert_calling_convention(func.sig_calling_convention())?,
            is_var_arg: func.sig_is_var_arg(),
            is_exported: func.sig_visibility() != llvm_ir::module::Visibility::Hidden,
            inline_hint: InlineHint::Auto,
            visibility: convert_visibility(func.sig_visibility()),
            attrs: FnAttrs {
                flags: FnAttrFlags::empty(),
                alignment: if func.sig_alignment() > 0 {
                    Some(func.sig_alignment())
                } else {
                    None
                },
                section: None,
                memory: None,
                string_attrs: Vec::new(),
            },
            dll_storage_class: convert_dll_storage_class(func.sig_dll_storage_class()),
        };
        convert_fn_attributes(
            func.sig_attributes(),
            &mut decl.attrs,
            &mut decl.inline_hint,
        );
        Ok(decl)
    }

    /// Convert an external function DECLARATION. The exhaustive
    /// destructure is the parse-side completeness contract: a new llvm-ir
    /// field breaks compilation here until it is carried or deliberately
    /// dropped.
    fn from_declaration(
        decl: &FunctionDeclaration,
        types: &mut TypeArena,
        module: &Module,
    ) -> color_eyre::Result<Self> {
        let FunctionDeclaration {
            // Converted by from_signature through the FnSignature trait.
            name: _,
            parameters: _,
            is_var_arg: _,
            return_type: _,
            return_attributes: _,
            linkage: _,
            visibility: _,
            dll_storage_class: _,
            calling_convention: _,
            alignment: _,
            function_attributes: _,
            garbage_collector_name,
            // No debug info support yet.
            debugloc: _debugloc,
        } = decl;
        if let Some(gc) = garbage_collector_name {
            return Err(eyre!(
                "function `{}` uses garbage collection strategy {gc:?}, which is not supported",
                decl.name
            ));
        }
        Self::from_signature(decl, types, module)
    }

    fn from_fn(
        func: &llvm_ir::Function,
        types: &mut TypeArena,
        module: &Module,
    ) -> color_eyre::Result<Self> {
        // Exhaustive destructure: the parse-side completeness contract for
        // function DEFINITIONS (see from_declaration for the rationale).
        let llvm_ir::Function {
            // Converted by from_signature through the FnSignature trait.
            name: _,
            parameters: _,
            is_var_arg: _,
            return_type: _,
            return_attributes: _,
            linkage: _,
            visibility: _,
            dll_storage_class: _,
            calling_convention: _,
            alignment: _,
            function_attributes: _,
            // Lowered by lower_fn_body after declaration.
            basic_blocks: _,
            // Converted below.
            section: _,
            // COMDAT selection only matters for C++ ODR-style link dedup;
            // linkonce/weak linkage covers the C corpus. Dropped until a
            // real input needs it.
            comdat: _comdat,
            garbage_collector_name,
            // An exception personality is only meaningful together with
            // invoke/landingpad, which the parser rejects outright; with no
            // invokes in the module the personality can never run, so
            // dropping it is safe.
            personality_function: _personality_function,
            // No debug info support yet.
            debugloc: _debugloc,
        } = func;
        if let Some(gc) = garbage_collector_name {
            return Err(eyre!(
                "function `{}` uses garbage collection strategy {gc:?}, which is not supported",
                func.name
            ));
        }

        // Read the signature fields directly off the `Function` (it and
        // `FunctionDeclaration` both implement `FnSignature`) -- no
        // temporary clone -- then layer on the definition-only fields.
        let mut decl = Self::from_signature(func, types, module)?;

        if let Some(section) = &func.section {
            decl.attrs.section = Some(section.clone());
        }

        Ok(decl)
    }
}

/// Convert the function-level attribute list into the RVSDG
/// representation, shared by definitions and declarations -- a
/// declaration's attributes are promises about the unseen body (memory
/// behaviour above all), which summary propagation seeds from.
fn convert_fn_attributes(
    source: &[llvm_ir::function::FunctionAttribute],
    attrs: &mut FnAttrs,
    inline_hint: &mut InlineHint,
) {
    use llvm_ir::function::FunctionAttribute;
    let none = MemoryEffects {
        other: ModRef::NoModRef,
        arg_mem: ModRef::NoModRef,
        inaccessible_mem: ModRef::NoModRef,
        errno_mem: ModRef::NoModRef,
    };
    for attr in source {
        match attr {
            FunctionAttribute::NoReturn => attrs.flags |= FnAttrFlags::NO_RETURN,
            FunctionAttribute::NoUnwind => attrs.flags |= FnAttrFlags::NO_UNWIND,
            FunctionAttribute::NoRecurse => attrs.flags |= FnAttrFlags::NO_RECURSE,
            FunctionAttribute::Cold => attrs.flags |= FnAttrFlags::COLD,
            FunctionAttribute::StackProtect => attrs.flags |= FnAttrFlags::STACK_PROTECT,
            FunctionAttribute::StackProtectReq => {
                attrs.flags |= FnAttrFlags::STACK_PROTECT_REQ;
            }
            FunctionAttribute::StackProtectStrong => {
                attrs.flags |= FnAttrFlags::STACK_PROTECT_STRONG;
            }
            FunctionAttribute::UWTable => attrs.flags |= FnAttrFlags::UWTABLE,
            FunctionAttribute::WillReturn => attrs.flags |= FnAttrFlags::WILL_RETURN,
            FunctionAttribute::NoSync => attrs.flags |= FnAttrFlags::NO_SYNC,
            FunctionAttribute::NoFree => attrs.flags |= FnAttrFlags::NO_FREE,
            FunctionAttribute::NoInline => {
                attrs.flags |= FnAttrFlags::NO_INLINE;
                *inline_hint = InlineHint::Never;
            }
            FunctionAttribute::AlwaysInline => {
                attrs.flags |= FnAttrFlags::ALWAYS_INLINE;
                *inline_hint = InlineHint::Always;
            }
            // Memory behaviour: LLVM 16+ input carries the composite
            // memory(...) attribute; the bare variants are its pre-16
            // spellings, mapped onto the same structure.
            FunctionAttribute::Memory {
                default,
                argmem,
                inaccessible_mem,
                errno_mem,
            } => {
                let conv = |e: &llvm_ir::function::MemoryEffect| match e {
                    llvm_ir::function::MemoryEffect::None => ModRef::NoModRef,
                    llvm_ir::function::MemoryEffect::Read => ModRef::Ref,
                    llvm_ir::function::MemoryEffect::Write => ModRef::Mod,
                    llvm_ir::function::MemoryEffect::ReadWrite => ModRef::ModRef,
                };
                attrs.memory = Some(MemoryEffects {
                    other: conv(default),
                    arg_mem: conv(argmem),
                    inaccessible_mem: conv(inaccessible_mem),
                    errno_mem: conv(errno_mem),
                });
            }
            FunctionAttribute::ReadNone => attrs.memory = Some(none),
            FunctionAttribute::ReadOnly => {
                attrs.memory = Some(MemoryEffects {
                    other: ModRef::Ref,
                    arg_mem: ModRef::Ref,
                    inaccessible_mem: ModRef::Ref,
                    errno_mem: ModRef::Ref,
                });
            }
            FunctionAttribute::WriteOnly => {
                attrs.memory = Some(MemoryEffects {
                    other: ModRef::Mod,
                    arg_mem: ModRef::Mod,
                    inaccessible_mem: ModRef::Mod,
                    errno_mem: ModRef::Mod,
                });
            }
            FunctionAttribute::ArgMemOnly => {
                attrs.memory = Some(MemoryEffects {
                    arg_mem: ModRef::ModRef,
                    ..none
                });
            }
            FunctionAttribute::InaccessibleMemOnly => {
                attrs.memory = Some(MemoryEffects {
                    inaccessible_mem: ModRef::ModRef,
                    ..none
                });
            }
            FunctionAttribute::InaccessibleMemOrArgMemOnly => {
                attrs.memory = Some(MemoryEffects {
                    arg_mem: ModRef::ModRef,
                    inaccessible_mem: ModRef::ModRef,
                    ..none
                });
            }
            // Carried verbatim: these steer codegen (target features,
            // stack-protector sizing, frame pointer policy, ...).
            FunctionAttribute::StringAttribute { kind, value } => {
                attrs.string_attrs.push((kind.clone(), value.clone()));
            }
            // Not modeled yet; the fidelity net surfaces any of these
            // the moment a real input carries them.
            _ => {}
        }
    }
}

fn convert_dll_storage_class(class: llvm_ir::module::DLLStorageClass) -> DllStorageClass {
    match class {
        llvm_ir::module::DLLStorageClass::Default => DllStorageClass::Default,
        llvm_ir::module::DLLStorageClass::Import => DllStorageClass::Import,
        llvm_ir::module::DLLStorageClass::Export => DllStorageClass::Export,
    }
}

fn convert_thread_local_mode(mode: llvm_ir::module::ThreadLocalMode) -> ThreadLocalMode {
    match mode {
        llvm_ir::module::ThreadLocalMode::NotThreadLocal => ThreadLocalMode::NotThreadLocal,
        llvm_ir::module::ThreadLocalMode::GeneralDynamic => ThreadLocalMode::GeneralDynamic,
        llvm_ir::module::ThreadLocalMode::LocalDynamic => ThreadLocalMode::LocalDynamic,
        llvm_ir::module::ThreadLocalMode::InitialExec => ThreadLocalMode::InitialExec,
        llvm_ir::module::ThreadLocalMode::LocalExec => ThreadLocalMode::LocalExec,
    }
}

fn convert_linkage(linkage: llvm_ir::module::Linkage) -> Linkage {
    match linkage {
        llvm_ir::module::Linkage::Private => Linkage::Private,
        llvm_ir::module::Linkage::Internal => Linkage::Internal,
        llvm_ir::module::Linkage::External | llvm_ir::module::Linkage::ExternalWeak => {
            Linkage::External
        }
        llvm_ir::module::Linkage::AvailableExternally => Linkage::AvailableExternally,
        llvm_ir::module::Linkage::LinkOnceAny | llvm_ir::module::Linkage::LinkOnceODRAutoHide => {
            Linkage::LinkOnce
        }
        llvm_ir::module::Linkage::LinkOnceODR => Linkage::LinkOnceODR,
        llvm_ir::module::Linkage::WeakAny => Linkage::Weak,
        llvm_ir::module::Linkage::WeakODR => Linkage::WeakODR,
        other => {
            todo!("handle linkage type: {other:?}");
        }
    }
}

fn convert_calling_convention(
    cc: llvm_ir::function::CallingConvention,
) -> color_eyre::Result<CallingConvention> {
    // Only conventions with an exact RVSDG counterpart are accepted. An unknown
    // or unmappable convention is rejected rather than silently remapped (e.g.
    // to C) -- remapping a calling convention is a silent miscompile, which is
    // exactly the kind of bug differential testing must not paper over.
    let converted = match cc {
        llvm_ir::function::CallingConvention::C => CallingConvention::C,
        llvm_ir::function::CallingConvention::Fast => CallingConvention::Fast,
        llvm_ir::function::CallingConvention::Cold => CallingConvention::Cold,
        llvm_ir::function::CallingConvention::GHC => CallingConvention::GHC,
        llvm_ir::function::CallingConvention::HiPE => CallingConvention::HiPE,
        llvm_ir::function::CallingConvention::PreserveMost => CallingConvention::PreserveMost,
        llvm_ir::function::CallingConvention::PreserveAll => CallingConvention::PreserveAll,
        llvm_ir::function::CallingConvention::Swift => CallingConvention::Swift,
        other => {
            return Err(color_eyre::eyre::eyre!(
                "unsupported calling convention: {other:?}"
            ));
        }
    };
    Ok(converted)
}

fn convert_visibility(vis: llvm_ir::module::Visibility) -> Visibility {
    match vis {
        llvm_ir::module::Visibility::Default => Visibility::Default,
        llvm_ir::module::Visibility::Hidden => Visibility::Hidden,
        llvm_ir::module::Visibility::Protected => Visibility::Protected,
    }
}

/// Sign-extend a zero-extended LLVM integer constant to i64.
///
/// LLVM stores integer constants as u64 with the value zero-extended to 64 bits.
/// Our IR stores them as i64. For the bit pattern to round-trip correctly through
/// `const_int(*val as u64, false)`, we need to sign-extend from the original
/// width so that negative values are represented correctly in the i64.
///
/// Examples:
///   - i8 `-1`:  LLVM stores 0xFF (255). Sign-extend -> -1i64. Lowering: -1i64 as u64 = 0xFFFFFFFF_FFFFFFFF, truncated to i8 = 0xFF. OK
///   - i8 `127`: LLVM stores 0x7F (127). Sign-extend -> 127i64. OK
///   - i32 `-1`: LLVM stores 0xFFFFFFFF. Sign-extend -> -1i64. OK
///   - i64 `-1`: LLVM stores 0xFFFFFFFF_FFFFFFFF. Cast -> -1i64. OK
pub(super) fn sign_extend_to_i64(value: u64, bits: u32) -> i64 {
    if bits >= 64 {
        value as i64
    } else {
        let shift = 64 - bits;
        ((value as i64) << shift) >> shift
    }
}

pub(super) fn int_bit_to_scalar(bits: u32) -> color_eyre::Result<ScalarType> {
    Ok(match bits {
        1 => ScalarType::Bool,
        8 => ScalarType::I8,
        16 => ScalarType::I16,
        32 => ScalarType::I32,
        64 => ScalarType::I64,
        128 => ScalarType::I128,
        // Bitfield storage units and other odd widths (i24, i40, ...).
        // Named widths were matched above, so IntArbitrary can never
        // duplicate them. Above 64 (except the named 128) stays
        // unsupported: ConstValue::Int carries an i64.
        2..=63 => ScalarType::IntArbitrary(bits as u16),
        _ => Err(eyre!(
            "unsupported integer width: {bits} (wider than 64 and not 128)"
        ))?,
    })
}

/// `Name::Display` prepends `%` (correct for SSA-locals, wrong for globals).
/// `fn_map` and `global_map` are keyed by the bare name, so this is the
/// canonical conversion to use whenever we need to insert into or look up
/// from those maps.
pub(super) fn global_name_string(name: &llvm_ir::Name) -> String {
    match name {
        llvm_ir::Name::Name(s) => s.as_ref().clone(),
        llvm_ir::Name::Number(n) => n.to_string(),
    }
}
