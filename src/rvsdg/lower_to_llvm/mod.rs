use crate::rvsdg::{
    FuncId, GlobalId, GlobalInit, Linkage, RVSDGMod, Region, ThreadLocalMode, ValueId, ValueKind,
    Visibility,
    func::{
        CallingConvention, FnAttrFlags, FnAttrs, Function, MemoryEffects, ModRef, ParamAttrFlags,
        ParamAttrsExtra, Signature,
    },
    global::{DllStorageClass, GlobalDef, UnnamedAddr},
    types::{ScalarType, TypeRef},
};
use color_eyre::eyre::{bail, eyre};
use inkwell::{
    AddressSpace, DLLStorageClass, GlobalVisibility, OptimizationLevel,
    attributes::{Attribute, AttributeLoc},
    builder::Builder,
    context::Context,
    module::Module,
    targets::{CodeModel, FileType, RelocMode, Target, TargetTriple},
    types::{AnyType, BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
    values::{
        BasicValue, BasicValueEnum, CallSiteValue, FunctionValue, GlobalValue, UnnamedAddress,
    },
};
use std::{path::Path, process::Command};

pub mod binary;
pub mod cast;
pub mod const_val;
pub mod gamma;
pub mod intrinsic;
pub mod memory;
pub mod test_utils;
pub mod theta;
pub mod unary;
pub mod value;

#[derive(Debug)]
pub struct LLVMBuilderCtx<'a, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    builder: &'a Builder<'ctx>,
}

#[derive(Debug)]
pub struct ValueMapper<'ctx> {
    values: Vec<Option<BasicValueEnum<'ctx>>>,
    fns: Vec<Option<FunctionValue<'ctx>>>,
    globals: Vec<Option<GlobalValue<'ctx>>>,
    /// Control nodes (gamma/theta) currently being lowered. A node has no
    /// mapper entry until it completes, so a value inside a construct that
    /// (illegally) references the construct itself would re-enter its
    /// lowering and recurse forever; this turns that into an error naming
    /// the node.
    in_progress: rustc_hash::FxHashSet<ValueId>,
}

impl<'ctx> ValueMapper<'ctx> {
    fn new(rvsdg_mod: &RVSDGMod) -> Self {
        Self {
            values: vec![None; rvsdg_mod.values.len()],
            fns: vec![None; rvsdg_mod.functions.len()],
            globals: vec![None; rvsdg_mod.globals.len()],
            in_progress: rustc_hash::FxHashSet::default(),
        }
    }

    /// Mark a control node's lowering as started; errors on re-entry.
    pub(crate) fn begin_control(&mut self, value_id: ValueId) -> color_eyre::Result<()> {
        if !self.in_progress.insert(value_id) {
            bail!(
                "control node {value_id:?} re-entered its own lowering: a value \
                 inside the construct references the construct itself"
            );
        }
        Ok(())
    }

    pub(crate) fn finish_control(&mut self, value_id: ValueId) {
        self.in_progress.remove(&value_id);
    }

    fn get_val(&self, value_id: ValueId) -> &Option<BasicValueEnum<'ctx>> {
        &self.values[value_id.0 as usize]
    }
    fn set_val(&mut self, value_id: ValueId, value_enum: BasicValueEnum<'ctx>) {
        self.values[value_id.0 as usize] = Some(value_enum);
    }

    fn get_fn(&self, func_id: FuncId) -> &Option<FunctionValue<'ctx>> {
        &self.fns[func_id.0 as usize]
    }
    fn set_fn(&mut self, func_id: FuncId, func: FunctionValue<'ctx>) {
        self.fns[func_id.0 as usize] = Some(func);
    }

    fn get_global(&self, global_id: GlobalId) -> &Option<GlobalValue<'ctx>> {
        &self.globals[global_id.0 as usize]
    }
    fn set_global(&mut self, global_id: GlobalId, global_value: GlobalValue<'ctx>) {
        self.globals[global_id.0 as usize] = Some(global_value);
    }
}

impl RVSDGMod {
    /// Lower the RVSDG module into an LLVM module without emitting files.
    /// The caller owns the context and module lifetime.
    #[tracing::instrument(skip_all)]
    pub fn lower_to_llvm_module<'ctx>(
        &self,
        context: &'ctx Context,
    ) -> color_eyre::Result<Module<'ctx>> {
        let module = context.create_module(&self.mod_name);
        if !self.module_asm.is_empty() {
            module.set_inline_assembly(&self.module_asm);
        }
        let builder = context.create_builder();
        let llvm_builder = LLVMBuilderCtx {
            context,
            module: &module,
            builder: &builder,
        };
        let mut value_mapper = ValueMapper::new(self);
        self.lower_mod(&llvm_builder, &mut value_mapper)?;
        Ok(module)
    }

    pub fn output_with_llvm(
        &self,
        output: &str,
        link_inputs: &[String],
        link_args: &[String],
        include_dirs: &[String],
        defines: &[String],
        quiet: bool,
    ) -> color_eyre::Result<()> {
        // initialise things (guarded so concurrent callers don't race the
        // process-global target registry)
        crate::init_llvm_native()?;

        let context = Context::create();
        let module = self.lower_to_llvm_module(&context)?;
        if !quiet {
            eprintln!("LLVM IR:");
            eprintln!("{}", module.print_to_string().to_string());
        }

        // more output things
        let llvm_triple = TargetTriple::create(&self.target.to_string());
        let target = Target::from_triple(&llvm_triple)
            .map_err(|e| eyre!("failed to get target for triple {}: {e}", self.target))?;

        // This opt level is the LLVM *CodeGenOptLevel* only -- `write_to_file`
        // runs the backend pipeline (instruction selection, scheduling,
        // register allocation, machine-level opts), never the mid-level IR
        // optimizer (instcombine/GVN/LICM/vectorize), which is only run via a
        // PassBuilder pipeline we deliberately don't invoke. So the mid level
        // stays "-O0" (RVSDG owns those optimizations); `Aggressive` (-O3)
        // gives the greedy register allocator and aggressive instruction
        // selection/scheduling for the final code, not extra IR optimization.
        let machine = target
            .create_target_machine(
                &llvm_triple,
                "generic",
                "",
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| eyre!("failed to create target machine for triple {}", self.target))?;

        let obj_file = format!("{}.o", output);
        let obj_path = Path::new(&obj_file);
        machine
            .write_to_file(&module, FileType::Object, obj_path)
            .map_err(|e| eyre!("failed to write object file {}: {e}", obj_path.display()))?;

        // Status/diagnostics go to stderr so the compiler never writes to
        // stdout -- that belongs to the compiled program when it runs.
        eprintln!("Wrote object file: {}", obj_path.display());

        let obj_arg = obj_path
            .to_str()
            .ok_or_else(|| eyre!("object path {} is not valid UTF-8", obj_path.display()))?;

        // Link the compiled object together with any extra inputs (e.g. a
        // benchmark harness like PolyBench's `utilities/polybench.c`). `cc`
        // compiles any `.c` inputs and links everything; the `-I` paths and
        // `-D` defines are passed so those sources see the same headers and
        // configuration the primary input was compiled with (PolyBench's
        // timer, for one, is compiled out of polybench.c unless
        // POLYBENCH_TIME reaches it and then silently reports zeros). The
        // extra link arguments (e.g. `-lm`) go after every object so
        // library flags resolve the symbols those objects reference.
        let mut link = Command::new("cc");
        link.arg(obj_arg);
        for input in link_inputs {
            link.arg(input);
        }
        for arg in link_args {
            link.arg(arg);
        }
        for dir in include_dirs {
            link.arg("-I").arg(dir);
        }
        for define in defines {
            link.arg(format!("-D{define}"));
        }
        link.args(["-o", output]);
        let status = link
            .status()
            .map_err(|e| eyre!("failed to invoke linker (cc): {e}"))?;

        if !status.success() {
            bail!("linking failed with status: {status}");
        }
        eprintln!("Linked executable: ./{output}");
        eprintln!("Run it with:  ./{output}");
        Ok(())
    }

    fn lower_mod<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
    ) -> color_eyre::Result<()> {
        // For now we'll use a naive implementation that converts the RVSDG directly to llvm
        // without using predicates.
        // TODO: replace this implemenation with this https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43246.pdf

        // Function declarations must precede global initializers: an
        // initializer can take a function's address (a function-pointer
        // table), and that FuncAddr resolves through the mapper, which only
        // has the function once register_fn has declared it. Globals must in
        // turn precede function bodies, which reference them freely.
        for func in self.functions.iter() {
            self.register_fn(llvm_builder, mapper, func)?;
        }
        self.lower_globals(llvm_builder, mapper)?;
        for func in self.functions.iter() {
            if func.lambda_val.is_none() {
                continue; // declaration only, no body to lower
            }
            self.lower_fn(llvm_builder, mapper, func)?;
        }
        if let Err(e) = llvm_builder.module.verify() {
            bail!("LLVM module verification failed: {e}");
        }

        Ok(())
    }

    fn lower_globals<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
    ) -> color_eyre::Result<()> {
        // Pass 1: declare every global. This must finish before any
        // initializer is lowered, because an initializer can reference another
        // global that is declared later (e.g. `@a = ptr @b` with `@b` below
        // `@a`); that GlobalAddr resolves through the mapper, which only has
        // the global once it's declared here. Same two-pass shape as the
        // frontend's `from_llvm_mod`.
        for (i, global) in self.globals.iter().enumerate() {
            let llvm_type = self.type_to_basic_type_llvm(llvm_builder.context, global.ty)?;
            let space = (global.addr_space != 0)
                .then(|| {
                    AddressSpace::try_from(global.addr_space).map_err(|_| {
                        eyre!(
                            "global {} has address space {} outside LLVM's range",
                            global.name,
                            global.addr_space
                        )
                    })
                })
                .transpose()?;
            let glob = llvm_builder
                .module
                .add_global(llvm_type, space, &global.name);
            apply_global_attrs(glob, global);
            mapper.set_global(GlobalId(i as u32), glob);
        }

        // Pass 2: set initializers, now that every global resolves.
        for (i, global) in self.globals.iter().enumerate() {
            if let GlobalInit::Init(const_id) = global.initializer {
                let const_val = self.lower_const_id(llvm_builder, mapper, const_id)?;
                mapper
                    .get_global(GlobalId(i as u32))
                    .ok_or_else(|| eyre!("global {i} was not declared in pass 1"))?
                    .set_initializer(&const_val as &dyn BasicValue);
            }
        }
        Ok(())
    }

    fn register_fn<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
    ) -> color_eyre::Result<()> {
        if rvsdg_func.return_types.len() >= 2 {
            bail!(
                "function `{}` has {} return values; LLVM supports at most one",
                rvsdg_func.name,
                rvsdg_func.return_types.len()
            );
        }

        let param_types = rvsdg_func
            .params
            .iter()
            .map(|param| self.type_to_basic_meta_llvm(llvm_builder.context, param.ty))
            .collect::<color_eyre::Result<Vec<_>>>()?;
        let llvm_fn_type = if let Some(&ret_ty) = rvsdg_func.return_types.first() {
            self.type_to_basic_type_llvm(llvm_builder.context, ret_ty)?
                .fn_type(&param_types, rvsdg_func.is_var_arg)
        } else {
            llvm_builder
                .context
                .void_type()
                .fn_type(&param_types, rvsdg_func.is_var_arg)
        };

        let func_ty = llvm_builder.module.add_function(
            &rvsdg_func.name,
            llvm_fn_type,
            Some(rvsdg_func.linkage_type.to_llvm()),
        );
        self.apply_function_abi(llvm_builder.context, func_ty, rvsdg_func)?;
        mapper.set_fn(rvsdg_func.id, func_ty);
        Ok(())
    }

    /// Re-apply a function's ABI to its LLVM declaration: calling
    /// convention plus the parameter and return attributes that change how
    /// values physically move (`byval` stack-copies, `sret` return slots,
    /// `zeroext`/`signext` extensions). Dropping any of these silently
    /// miscompiles calls across an externally-compiled boundary.
    fn apply_function_abi<'ctx>(
        &self,
        context: &'ctx Context,
        func_value: FunctionValue<'ctx>,
        function: &Function,
    ) -> color_eyre::Result<()> {
        func_value.set_call_conventions(calling_convention_to_llvm(function.calling_convention));
        let mut attributes = Vec::new();
        for (index, param) in function.params.iter().enumerate() {
            self.collect_abi_attributes(
                context,
                AttributeLoc::Param(index as u32),
                param.flags,
                param.extra.as_deref(),
                &mut attributes,
            )?;
        }
        self.collect_abi_attributes(
            context,
            AttributeLoc::Return,
            function.return_attrs.flags,
            function.return_attrs.extra.as_deref(),
            &mut attributes,
        )?;
        for (loc, attribute) in attributes {
            func_value.add_attribute(loc, attribute);
        }

        // Function-level attributes. Each flag maps to its LLVM enum
        // attribute; uwtable takes a value (2 = async, the plain "uwtable"
        // spelling llvm-ir parses everything back to).
        const FLAG_ATTRIBUTES: &[(FnAttrFlags, &str, u64)] = &[
            (FnAttrFlags::NO_RETURN, "noreturn", 0),
            (FnAttrFlags::NO_UNWIND, "nounwind", 0),
            (FnAttrFlags::NO_RECURSE, "norecurse", 0),
            (FnAttrFlags::NO_INLINE, "noinline", 0),
            (FnAttrFlags::ALWAYS_INLINE, "alwaysinline", 0),
            (FnAttrFlags::COLD, "cold", 0),
            (FnAttrFlags::STACK_PROTECT, "ssp", 0),
            (FnAttrFlags::STACK_PROTECT_REQ, "sspreq", 0),
            (FnAttrFlags::STACK_PROTECT_STRONG, "sspstrong", 0),
            (FnAttrFlags::UWTABLE, "uwtable", 2),
            (FnAttrFlags::WILL_RETURN, "willreturn", 0),
            (FnAttrFlags::NO_SYNC, "nosync", 0),
            (FnAttrFlags::NO_FREE, "nofree", 0),
        ];
        let FnAttrs {
            flags,
            alignment,
            section,
            memory,
            string_attrs,
        } = &function.attrs;
        for &(flag, name, value) in FLAG_ATTRIBUTES {
            if flags.contains(flag) {
                func_value.add_attribute(
                    AttributeLoc::Function,
                    context.create_enum_attribute(Attribute::get_named_enum_kind_id(name), value),
                );
            }
        }
        if let Some(memory) = memory {
            func_value.add_attribute(
                AttributeLoc::Function,
                context.create_enum_attribute(
                    Attribute::get_named_enum_kind_id("memory"),
                    memory_effects_to_llvm(*memory),
                ),
            );
        }
        for (kind, value) in string_attrs {
            func_value.add_attribute(
                AttributeLoc::Function,
                context.create_string_attribute(kind, value),
            );
        }
        if let Some(align) = alignment {
            func_value.as_global_value().set_alignment(*align);
        }
        if let Some(section) = section {
            func_value.as_global_value().set_section(Some(section));
        }
        // LLVM rejects non-default visibility on local linkage; only apply
        // a real visibility.
        match function.visibility {
            Visibility::Default => {}
            Visibility::Hidden => func_value
                .as_global_value()
                .set_visibility(GlobalVisibility::Hidden),
            Visibility::Protected => func_value
                .as_global_value()
                .set_visibility(GlobalVisibility::Protected),
        }
        match function.dll_storage_class {
            DllStorageClass::Default => {}
            DllStorageClass::Import => func_value
                .as_global_value()
                .set_dll_storage_class(DLLStorageClass::Import),
            DllStorageClass::Export => func_value
                .as_global_value()
                .set_dll_storage_class(DLLStorageClass::Export),
        }
        Ok(())
    }

    /// Re-apply a call site's ABI from its interned signature: calling
    /// convention, one attribute set per ACTUAL argument, return
    /// attributes. LLVM attributes live on call sites as well as
    /// declarations (a declared `byval` means nothing to codegen unless
    /// the call site carries it too), and the signature is the only
    /// place the full site ABI exists: indirect callees are opaque
    /// pointers, and a variadic call's trailing arguments have no
    /// declaration entries at all. Used for BOTH direct and indirect
    /// calls -- one implementation, no drift.
    pub(crate) fn apply_call_site_abi<'ctx>(
        &self,
        context: &'ctx Context,
        call_site: CallSiteValue<'ctx>,
        signature: &Signature,
    ) -> color_eyre::Result<()> {
        call_site.set_call_convention(calling_convention_to_llvm(signature.calling_convention));
        let mut attributes = Vec::new();
        for (index, attrs) in signature.param_attrs.iter().enumerate() {
            self.collect_abi_attributes(
                context,
                AttributeLoc::Param(index as u32),
                attrs.flags,
                attrs.extra.as_deref(),
                &mut attributes,
            )?;
        }
        self.collect_abi_attributes(
            context,
            AttributeLoc::Return,
            signature.return_attrs.flags,
            signature.return_attrs.extra.as_deref(),
            &mut attributes,
        )?;
        for (loc, attribute) in attributes {
            call_site.add_attribute(loc, attribute);
        }
        Ok(())
    }

    /// The attributes of one parameter or return slot, as inkwell
    /// attributes: the ABI-bearing ones (zeroext/signext here,
    /// byval/sret/align in `extra`) and the optimisation hints, which the
    /// fidelity net holds us to re-emitting faithfully.
    fn collect_abi_attributes<'ctx>(
        &self,
        context: &'ctx Context,
        loc: AttributeLoc,
        flags: ParamAttrFlags,
        extra: Option<&ParamAttrsExtra>,
        out: &mut Vec<(AttributeLoc, Attribute)>,
    ) -> color_eyre::Result<()> {
        const FLAG_ATTRIBUTES: &[(ParamAttrFlags, &str)] = &[
            (ParamAttrFlags::ZERO_EXTEND, "zeroext"),
            (ParamAttrFlags::SIGN_EXTEND, "signext"),
            (ParamAttrFlags::NO_ALIAS, "noalias"),
            (ParamAttrFlags::NO_CAPTURE, "nocapture"),
            (ParamAttrFlags::NON_NULL, "nonnull"),
            (ParamAttrFlags::READ_ONLY, "readonly"),
            (ParamAttrFlags::WRITE_ONLY, "writeonly"),
            (ParamAttrFlags::NO_UNDEF, "noundef"),
            (ParamAttrFlags::RETURNED, "returned"),
        ];
        for &(flag, name) in FLAG_ATTRIBUTES {
            if flags.contains(flag) {
                out.push((
                    loc,
                    context.create_enum_attribute(Attribute::get_named_enum_kind_id(name), 0),
                ));
            }
        }
        if let Some(extra) = extra {
            if let Some(ty) = extra.by_value {
                let llvm_ty = self.type_to_basic_type_llvm(context, ty)?;
                out.push((
                    loc,
                    context.create_type_attribute(
                        Attribute::get_named_enum_kind_id("byval"),
                        llvm_ty.as_any_type_enum(),
                    ),
                ));
            }
            if let Some(ty) = extra.struct_return {
                let llvm_ty = self.type_to_basic_type_llvm(context, ty)?;
                out.push((
                    loc,
                    context.create_type_attribute(
                        Attribute::get_named_enum_kind_id("sret"),
                        llvm_ty.as_any_type_enum(),
                    ),
                ));
            }
            if let Some(align) = extra.alignment {
                out.push((
                    loc,
                    context.create_enum_attribute(
                        Attribute::get_named_enum_kind_id("align"),
                        u64::from(align),
                    ),
                ));
            }
            if let Some(bytes) = extra.dereferenceable_bytes {
                out.push((
                    loc,
                    context.create_enum_attribute(
                        Attribute::get_named_enum_kind_id("dereferenceable"),
                        bytes,
                    ),
                ));
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip_all, fields(func = %rvsdg_func.name))]
    fn lower_fn<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
    ) -> color_eyre::Result<()> {
        let func = mapper.get_fn(rvsdg_func.id).ok_or_else(|| {
            eyre!(
                "function `{}` was not registered before lowering its body",
                rvsdg_func.name
            )
        })?;
        let entry = llvm_builder.context.append_basic_block(func, "entry");
        llvm_builder.builder.position_at_end(entry);
        let fn_val = rvsdg_func.lambda_val.ok_or_else(|| {
            eyre!(
                "function `{}` has no lambda value set during RVSDG construction",
                rvsdg_func.name
            )
        })?;
        let lambda_val = &self.values[fn_val.0 as usize];
        match &lambda_val.kind {
            ValueKind::Lambda {
                region: region_id,
                func_id: _,
            } => {
                // register the regions inputs to the llvm functions parameters so that they can be
                // referenced by project inside the region
                let region = &self.regions[region_id.0 as usize];
                for (i, &param_id) in region.params.iter().enumerate() {
                    let param = func.get_nth_param(i as u32).ok_or_else(|| {
                        eyre!("function `{}` is missing parameter {i}", rvsdg_func.name)
                    })?;
                    mapper.set_val(param_id, param);
                }

                self.lower_region(llvm_builder, mapper, rvsdg_func, region)?;

                // regions results should be added from inside lower_region
                let res = self.value_pool.get(region.results);
                match res.len() {
                    0 => llvm_builder.builder.build_return(None)?,
                    1 => {
                        let val = self
                            .lowered_result(llvm_builder, mapper, rvsdg_func, res[0])?
                            .ok_or_else(|| {
                                eyre!("return value of `{}` was not lowered", rvsdg_func.name)
                            })?;
                        llvm_builder
                            .builder
                            .build_return(Some(&val as &dyn BasicValue))?
                    }
                    n => bail!(
                        "function `{}` returns {n} values; LLVM supports at most one",
                        rvsdg_func.name
                    ),
                }
            }
            t => bail!(
                "function `{}` lambda has unexpected value kind {t:?}",
                rvsdg_func.name
            ),
        };
        Ok(())
    }

    /// Lower every node of a region, in `region.nodes` order.
    ///
    /// This linear walk is what honours the STATE edges: construction
    /// appends nodes in symbolic-execution order, so the list is a
    /// topological order of the state chain, and the LLVM builder's
    /// insertion point only ever advances -- emission position IS the
    /// state order. That is why the per-kind lowering arms destructure
    /// `state: _`: LLVM has no state values (ordering between memory
    /// instructions is positional within a block), so there is nothing to
    /// resolve or return for a state edge at this level. Side-effecting
    /// nodes must only ever be lowered by this walk, never on demand.
    fn lower_region<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        region: &Region,
    ) -> color_eyre::Result<()> {
        for &value_id in region.nodes.iter() {
            self.lower_value(llvm_builder, mapper, rvsdg_func, value_id)?;
        }
        Ok(())
    }
    fn type_to_basic_type_llvm<'b>(
        &self,
        context: &'b Context,
        ty: TypeRef,
    ) -> color_eyre::Result<BasicTypeEnum<'b>> {
        let basic = match ty {
            TypeRef::State => bail!("`state` is an IR-only type with no LLVM basic type"),
            TypeRef::Scalar(scalar_type) => match scalar_type {
                ScalarType::Bool => BasicTypeEnum::IntType(context.bool_type()),
                ScalarType::I8 => BasicTypeEnum::IntType(context.i8_type()),
                ScalarType::I16 => BasicTypeEnum::IntType(context.i16_type()),
                ScalarType::I32 => BasicTypeEnum::IntType(context.i32_type()),
                ScalarType::I64 => BasicTypeEnum::IntType(context.i64_type()),
                ScalarType::I128 => BasicTypeEnum::IntType(context.i128_type()),
                ScalarType::IntArbitrary(bits) => BasicTypeEnum::IntType(
                    context
                        .custom_width_int_type(
                            std::num::NonZeroU32::new(bits as u32)
                                .ok_or_else(|| eyre!("zero-width integer type"))?,
                        )
                        .map_err(|e| eyre!("invalid integer width {bits}: {e}"))?,
                ),
                ScalarType::F32 => BasicTypeEnum::FloatType(context.f32_type()),
                ScalarType::F64 => BasicTypeEnum::FloatType(context.f64_type()),
                ScalarType::F80 => BasicTypeEnum::FloatType(context.x86_f80_type()),
                // Void is not a BasicType in LLVM -- it only appears as a function
                // return type, never as a value/parameter/alloca type.
                ScalarType::Void => bail!("`void` is not a basic type"),
            },
            TypeRef::Ptr(_) => {
                BasicTypeEnum::PointerType(context.ptr_type(AddressSpace::default()))
            }
            TypeRef::Array(array_type_id) => {
                let arr = self.types.get_array(array_type_id);
                let elem = self.type_to_basic_type_llvm(context, arr.element)?;
                BasicTypeEnum::ArrayType(elem.array_type(arr.len as u32))
            }
            TypeRef::Struct(struct_id) => {
                let def = self.types.get_struct(struct_id);
                let field_types: Vec<BasicTypeEnum> = def
                    .fields
                    .iter()
                    .map(|f| self.type_to_basic_type_llvm(context, f.field_type))
                    .collect::<color_eyre::Result<_>>()?;
                let llvm_struct = match &def.name {
                    // Named structs are identity-based in LLVM and live in
                    // the context: look the name up first so repeated
                    // conversions reuse ONE %name instead of minting
                    // %name.0, %name.1, ... per call.
                    Some(name) => match context.get_struct_type(name) {
                        Some(existing) => existing,
                        None => {
                            let created = context.opaque_struct_type(name);
                            created.set_body(&field_types, def.packed);
                            created
                        }
                    },
                    None => context.struct_type(&field_types, def.packed),
                };
                BasicTypeEnum::StructType(llvm_struct)
            }
            TypeRef::Vector(vector_type_id) => {
                let vec = self.types.get_vector(vector_type_id);
                let elem = self.type_to_basic_type_llvm(context, vec.element)?;
                match elem {
                    BasicTypeEnum::IntType(t) => BasicTypeEnum::VectorType(t.vec_type(vec.lanes)),
                    BasicTypeEnum::FloatType(t) => BasicTypeEnum::VectorType(t.vec_type(vec.lanes)),
                    BasicTypeEnum::PointerType(t) => {
                        BasicTypeEnum::VectorType(t.vec_type(vec.lanes))
                    }
                    _ => bail!("vector element must be a scalar or pointer type"),
                }
            }
            // FuncType is not a BasicType -- functions exist only as pointers
            // (opaque ptr in LLVM 17+). If a TypeRef::Func reaches here, the
            // caller has a bug.
            TypeRef::Func(_) => bail!("function type is not a basic type"),
            // A control/predicate value is an alternative index; lower it to an
            // `i32`. This is what makes the gamma backend take its switch path
            // (value k -> region k) for predicate-driven gammas.
            TypeRef::Control(_) => BasicTypeEnum::IntType(context.i32_type()),
        };
        Ok(basic)
    }

    /// A parameter/argument-position type. Identical to
    /// [`type_to_basic_type_llvm`](Self::type_to_basic_type_llvm) except for the
    /// wrapper enum, so it just converts that result -- every `BasicTypeEnum`
    /// has a `BasicMetadataTypeEnum` counterpart.
    fn type_to_basic_meta_llvm<'b>(
        &self,
        context: &'b Context,
        ty: TypeRef,
    ) -> color_eyre::Result<BasicMetadataTypeEnum<'b>> {
        Ok(self.type_to_basic_type_llvm(context, ty)?.into())
    }
}

/// Encode memory effects as LLVM's composite memory(...) attribute value.
///
/// Unlike the calling-convention numbers there is no library source to
/// borrow this from: LLVM's C API takes the raw u64 with no constructor,
/// inkwell adds nothing, and llvm-ir's decoder is crate-private. The
/// layout (two may-read/may-write bits per location; argmem at bits 0-1,
/// inaccessible memory at 2-3, everything else at 4-5; read is 0b01,
/// write 0b10) mirrors LLVM's ModRef.h and llvm-ir's decoder of the same
/// value. It cannot silently drift: the fidelity tests round-trip
/// memory(none) and memory(read) functions through LLVM's printer and
/// llvm-ir's decoder, so a wrong encoding fails those tests.
fn memory_effects_to_llvm(memory: MemoryEffects) -> u64 {
    let bits = |m: ModRef| -> u64 {
        match m {
            ModRef::NoModRef => 0b00,
            ModRef::Ref => 0b01,
            ModRef::Mod => 0b10,
            ModRef::ModRef => 0b11,
        }
    };
    bits(memory.arg_mem) | (bits(memory.inaccessible_mem) << 2) | (bits(memory.other) << 4)
}

/// The numeric LLVM calling-convention id. The numbers come from LLVM's
/// own C header via llvm-sys's `LLVMCallConv` (re-exported through
/// inkwell), so they can never drift from the linked LLVM; inkwell's
/// setters take the raw number.
fn calling_convention_to_llvm(cc: CallingConvention) -> u32 {
    use inkwell::llvm_sys::LLVMCallConv;
    let conv = match cc {
        CallingConvention::C => LLVMCallConv::LLVMCCallConv,
        CallingConvention::Fast => LLVMCallConv::LLVMFastCallConv,
        CallingConvention::Cold => LLVMCallConv::LLVMColdCallConv,
        CallingConvention::GHC => LLVMCallConv::LLVMGHCCallConv,
        CallingConvention::HiPE => LLVMCallConv::LLVMHiPECallConv,
        CallingConvention::PreserveMost => LLVMCallConv::LLVMPreserveMostCallConv,
        CallingConvention::PreserveAll => LLVMCallConv::LLVMPreserveAllCallConv,
        CallingConvention::Swift => LLVMCallConv::LLVMSwiftCallConv,
        CallingConvention::X86StdCall => LLVMCallConv::LLVMX86StdcallCallConv,
        CallingConvention::X86FastCall => LLVMCallConv::LLVMX86FastcallCallConv,
        CallingConvention::ArmAAPCS => LLVMCallConv::LLVMARMAAPCSCallConv,
        CallingConvention::ArmAAPCSVFP => LLVMCallConv::LLVMARMAAPCSVFPCallConv,
        CallingConvention::X86ThisCall => LLVMCallConv::LLVMX86ThisCallCallConv,
        CallingConvention::X86_64SysV => LLVMCallConv::LLVMX8664SysVCallConv,
        CallingConvention::Win64 => LLVMCallConv::LLVMWin64CallConv,
        CallingConvention::X86VectorCall => LLVMCallConv::LLVMX86VectorCallCallConv,
        CallingConvention::X86RegCall => LLVMCallConv::LLVMX86RegCallCallConv,
        CallingConvention::Numbered(n) => return n,
    };
    conv as u32
}

/// Re-apply every attribute a [`GlobalDef`] carries to its emitted LLVM
/// global. The exhaustive destructure is the emit-side completeness
/// contract: adding a field to `GlobalDef` breaks compilation HERE until
/// it is applied or deliberately ignored.
fn apply_global_attrs(glob: GlobalValue, def: &GlobalDef) {
    let GlobalDef {
        // Consumed at creation (add_global).
        name: _name,
        ty: _ty,
        addr_space: _addr_space,
        // Applied by pass 2 of lower_globals, once every global resolves.
        initializer: _initializer,
        is_constant,
        linkage,
        alignment,
        section,
        visibility,
        thread_local,
        unnamed_addr,
        dll_storage_class,
    } = def;

    glob.set_constant(*is_constant);
    glob.set_linkage(linkage.to_llvm());
    if let Some(align) = alignment {
        glob.set_alignment(*align);
    }
    if let Some(section) = section {
        glob.set_section(Some(section));
    }
    // LLVM rejects non-default visibility on local linkage, so only apply
    // a real visibility (the default needs no call anyway).
    match visibility {
        Visibility::Default => {}
        Visibility::Hidden => glob.set_visibility(GlobalVisibility::Hidden),
        Visibility::Protected => glob.set_visibility(GlobalVisibility::Protected),
    }
    // Thread-locality must survive re-emission: accesses go through the
    // llvm.threadlocal.address intrinsic, and LLVM verification requires
    // its operand to actually be thread-local.
    glob.set_thread_local_mode(thread_local_mode_to_llvm(*thread_local));
    match unnamed_addr {
        UnnamedAddr::None => {}
        UnnamedAddr::Local => glob.set_unnamed_address(UnnamedAddress::Local),
        UnnamedAddr::Global => glob.set_unnamed_address(UnnamedAddress::Global),
    }
    match dll_storage_class {
        DllStorageClass::Default => {}
        DllStorageClass::Import => glob.set_dll_storage_class(DLLStorageClass::Import),
        DllStorageClass::Export => glob.set_dll_storage_class(DLLStorageClass::Export),
    }
}

/// Map the RVSDG thread-local mode to inkwell's; `None` means not
/// thread-local.
fn thread_local_mode_to_llvm(mode: ThreadLocalMode) -> Option<inkwell::ThreadLocalMode> {
    match mode {
        ThreadLocalMode::NotThreadLocal => None,
        ThreadLocalMode::GeneralDynamic => Some(inkwell::ThreadLocalMode::GeneralDynamicTLSModel),
        ThreadLocalMode::LocalDynamic => Some(inkwell::ThreadLocalMode::LocalDynamicTLSModel),
        ThreadLocalMode::InitialExec => Some(inkwell::ThreadLocalMode::InitialExecTLSModel),
        ThreadLocalMode::LocalExec => Some(inkwell::ThreadLocalMode::LocalExecTLSModel),
    }
}

impl Linkage {
    fn to_llvm(&self) -> inkwell::module::Linkage {
        match self {
            Linkage::Private => inkwell::module::Linkage::Private,
            Linkage::Internal => inkwell::module::Linkage::Internal,
            Linkage::External => inkwell::module::Linkage::External,
            Linkage::LinkOnce => inkwell::module::Linkage::LinkOnceAny,
            Linkage::LinkOnceODR => inkwell::module::Linkage::LinkOnceODR,
            Linkage::Weak => inkwell::module::Linkage::WeakAny,
            Linkage::WeakODR => inkwell::module::Linkage::WeakODR,
            Linkage::AvailableExternally => inkwell::module::Linkage::AvailableExternally,
        }
    }
}
