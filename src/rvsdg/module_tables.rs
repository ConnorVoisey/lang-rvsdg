use rustc_hash::FxHashMap;

use crate::rvsdg::{
    ConstantPool, FuncId, GlobalDef, GlobalId, InlineHint, Linkage, Visibility,
    func::{
        CallingConvention, FnAttrs, FnDecl, Function, Param, ParamAttrFlags, ParamAttrs, Signature,
        SignatureTable,
    },
    global::DllStorageClass,
    types::{FuncType, TypeArena, TypeRef, VOID},
};

#[derive(Debug, Default)]
pub struct ModuleTables {
    pub types: TypeArena,
    pub functions: Vec<Function>,
    pub globals: Vec<GlobalDef>,
    // These maps should probably use &str instead of String
    pub fn_map: FxHashMap<String, FuncId>,
    pub global_map: FxHashMap<String, GlobalId>,

    /// Interned ABI signatures for indirect call sites (see
    /// [`func::Signature`]).
    pub signatures: SignatureTable,
    pub constants: ConstantPool,
}

impl ModuleTables {
    #[inline]
    pub fn get_function(&self, func_id: FuncId) -> &Function {
        &self.functions[func_id.0 as usize]
    }

    #[inline]
    pub fn get_function_by_name(&self, name: &str) -> Option<&Function> {
        self.fn_map.get(name).map(|v| self.get_function(*v))
    }

    /// Simple declaration with default attributes and C calling convention.
    /// See also [`RVSDGMod::declare_fn_full`]
    pub fn declare_fn(
        &mut self,
        name: String,
        params: &[TypeRef],
        ret_types: &[TypeRef],
        linkage_type: Linkage,
    ) -> FuncId {
        self.declare_fn_full(FnDecl {
            name,
            params: params
                .iter()
                .map(|&ty| Param {
                    ty,
                    flags: ParamAttrFlags::empty(),
                    extra: None,
                })
                .collect(),
            return_types: ret_types.to_vec(),
            return_attrs: ParamAttrs::default(),
            linkage_type,
            calling_convention: CallingConvention::default(),
            is_var_arg: false,
            is_exported: false,
            inline_hint: InlineHint::Auto,
            visibility: Visibility::default(),
            attrs: FnAttrs::default(),
            dll_storage_class: DllStorageClass::default(),
        })
    }

    /// Full declaration with explicit control over all function metadata.
    /// The declaration-verbatim call signature is interned HERE, once,
    /// so body construction (the future parallel phase) reads
    /// [`Function::declared_sig`] instead of interning through the
    /// tables.
    pub fn declare_fn_full(&mut self, decl: FnDecl) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        // FuncType models LLVM's single return slot; a multi-return
        // RVSDG function has no LLVM function type to describe it, so
        // its declared_sig is None and direct calls must supply a
        // site signature.
        let declared_sig = match decl.return_types.as_slice() {
            [] | [_] => {
                let func_type = self.types.intern_fn(FuncType {
                    params: decl.params.iter().map(|p| p.ty).collect(),
                    ret: decl.return_types.first().copied().unwrap_or(VOID),
                    is_var_arg: decl.is_var_arg,
                });
                Some(
                    self.signatures.intern(Signature {
                        func_type,
                        param_attrs: decl
                            .params
                            .iter()
                            .map(|p| ParamAttrs {
                                flags: p.flags,
                                extra: p.extra.clone(),
                            })
                            .collect(),
                        return_attrs: decl.return_attrs.clone(),
                        calling_convention: decl.calling_convention,
                    }),
                )
            }
            _ => None,
        };
        let func = Function {
            id,
            name: decl.name.clone(),
            params: decl.params,
            return_types: decl.return_types,
            return_attrs: decl.return_attrs,
            declared_sig,
            is_exported: decl.is_exported,
            inline_hint: decl.inline_hint,
            linkage_type: decl.linkage_type,
            calling_convention: decl.calling_convention,
            is_var_arg: decl.is_var_arg,
            visibility: decl.visibility,
            attrs: decl.attrs,
            dll_storage_class: decl.dll_storage_class,
        };
        self.functions.push(func);
        self.fn_map.insert(decl.name, id);
        id
    }
}
