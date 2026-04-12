use crate::rvsdg::{
    ConstId, ConstValue, ConstantDef, ConstantKind, GlobalInit, InlineHint, Linkage, RVSDGMod,
    Visibility,
    func::{
        CallingConvention, FnAttrFlags, FnAttrs, FnDecl, Function, Param, ParamAttrFlags,
        ParamAttrs,
    },
    types::{
        ArrayType, FuncType, PtrType, ScalarType, StructDef, StructField, TypeArena, TypeRef, VOID,
        VectorType,
    },
};
use color_eyre::eyre::eyre;
use llvm_ir::{ConstantRef, Module, TypeRef as LLVMTypeRef, function::FunctionDeclaration};
use std::str::FromStr;
use target_lexicon::Triple;

impl RVSDGMod {
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

        // lower function declerations
        for func in &module.func_declarations {
            let decl = FnDecl::from_fn_decleration(&func, &mut rvsdg_mod.types, &module)?;
            rvsdg_mod.declare_fn_full(decl);
        }
        for func in &module.functions {
            let decl = FnDecl::from_fn(&func, &mut rvsdg_mod.types, &module)?;
            rvsdg_mod.declare_fn_full(decl);
        }

        // lower globals
        for global in &module.global_vars {
            let ty = rvsdg_mod.types.convert_type_ref(&global.ty, &module)?;
            let ty = match ty {
                // not sure if this will work if we have a global that points to a value that
                // hasn't been declared yet
                TypeRef::Ptr(ptr_type_id) => rvsdg_mod.types.get_ptr(ptr_type_id).pointee.unwrap(),
                t => t,
            };
            let init = match &global.initializer {
                Some(v) => GlobalInit::Init(rvsdg_mod.convert_const_ref(v.clone(), &module)?),
                None => GlobalInit::Extern,
            };
            rvsdg_mod.define_global(
                global.name.to_string(),
                ty,
                init,
                global.is_constant,
                convert_linkage(global.linkage),
            );
        }
        for global in &module.global_aliases {
            let ty = rvsdg_mod.types.convert_type_ref(&global.ty, &module)?;
            let init =
                GlobalInit::Init(rvsdg_mod.convert_const_ref(global.aliasee.clone(), &module)?);
            rvsdg_mod.define_global(
                global.name.to_string(),
                ty,
                init,
                true,
                convert_linkage(global.linkage),
            );
        }
        for global in &module.global_ifuncs {
            let ty = rvsdg_mod.types.convert_type_ref(&global.ty, &module)?;
            rvsdg_mod.define_global(
                global.name.to_string(),
                ty,
                GlobalInit::Extern,
                true,
                convert_linkage(global.linkage),
            );
        }
        // TODO: lower types
        // pub types: Types,

        // lower function bodies
        for func in &module.functions {
            rvsdg_mod.lower_fn_body(&func);
        }

        Ok(rvsdg_mod)
    }
    fn convert_const_ref(
        &mut self,
        const_ref: ConstantRef,
        module: &Module,
    ) -> color_eyre::Result<ConstId> {
        let const_def = match &*const_ref {
            llvm_ir::Constant::Int { bits, value } => ConstantDef {
                ty: TypeRef::Scalar(int_bit_to_scalar(*bits)?),
                kind: ConstantKind::Scalar(ConstValue::Int(sign_extend_to_i64(*value, *bits))),
            },
            llvm_ir::Constant::Float(float) => ConstantDef {
                ty: TypeRef::Scalar(match float {
                    llvm_ir::constant::Float::Single(_) => ScalarType::F32,
                    llvm_ir::constant::Float::Double(_) => ScalarType::F64,
                    t => Err(eyre!("unsupported float width: {t:?}"))?,
                }),
                kind: ConstantKind::Scalar(match float {
                    llvm_ir::constant::Float::Single(v) => ConstValue::F32(v.to_bits()),
                    llvm_ir::constant::Float::Double(v) => ConstValue::F64(v.to_bits()),
                    t => Err(eyre!("unsupported float width: {t:?}"))?,
                }),
            },
            llvm_ir::Constant::Null(_type_ref) => ConstantDef {
                ty: TypeRef::Ptr(self.types.intern_ptr(PtrType {
                    pointee: None,
                    alias_set: None,
                    no_escape: false,
                })),
                kind: ConstantKind::Scalar(ConstValue::NullPtr),
            },
            llvm_ir::Constant::AggregateZero(type_ref) => {
                let ty = self.types.convert_type_ref(type_ref, module)?;
                ConstantDef {
                    ty,
                    kind: ConstantKind::Zero,
                }
            }
            llvm_ir::Constant::Struct {
                name,
                values,
                is_packed,
            } => todo!(),
            llvm_ir::Constant::Array {
                element_type,
                elements,
            } => todo!(),
            llvm_ir::Constant::Vector(constant_refs) => todo!(),
            llvm_ir::Constant::Undef(type_ref) => {
                let ty = self.types.convert_type_ref(type_ref, module)?;
                ConstantDef {
                    ty,
                    kind: ConstantKind::Undef,
                }
            }
            llvm_ir::Constant::Poison(type_ref) => {
                let ty = self.types.convert_type_ref(type_ref, module)?;
                ConstantDef {
                    ty,
                    kind: ConstantKind::Scalar(ConstValue::Poison),
                }
            }
            llvm_ir::Constant::BlockAddress => todo!(),
            llvm_ir::Constant::GlobalReference { name, ty } => {
                let name_str = name.to_string();
                let ty = self.types.convert_type_ref(ty, module)?;
                if let Some(&global_id) = self.global_map.get(&name_str) {
                    ConstantDef {
                        ty,
                        kind: ConstantKind::GlobalAddr(global_id),
                    }
                } else if let Some(&_func_id) = self.fn_map.get(&name_str) {
                    // TODO: add ConstantKind::FuncAddr(FuncId) for function pointer constants
                    todo!("function reference in constant context: {name_str}")
                } else {
                    return Err(eyre!("global reference to unknown symbol: {name_str}"));
                }
            }
            // TokenNone is used for convergence control tokens in LLVM IR.
            // It has no runtime value — treat as a zero-sized placeholder.
            llvm_ir::Constant::TokenNone => ConstantDef {
                ty: VOID,
                kind: ConstantKind::Zero,
            },

            // The RVSDG IR layer does not have a concept of constant expressions —
            // these are evaluated at parse time.
            // These are only for ints, and only for lower llvm versions
            llvm_ir::Constant::Add(op) => {
                self.fold_int_binop(&op.operand0, &op.operand1, module, i64::wrapping_add)?
            }
            llvm_ir::Constant::Sub(op) => {
                self.fold_int_binop(&op.operand0, &op.operand1, module, i64::wrapping_sub)?
            }
            llvm_ir::Constant::Mul(op) => {
                self.fold_int_binop(&op.operand0, &op.operand1, module, i64::wrapping_mul)?
            }
            llvm_ir::Constant::Xor(op) => {
                self.fold_int_binop(&op.operand0, &op.operand1, module, |a, b| a ^ b)?
            }
            llvm_ir::Constant::ExtractElement(extract_element) => todo!(),
            llvm_ir::Constant::InsertElement(insert_element) => todo!(),
            llvm_ir::Constant::ShuffleVector(shuffle_vector) => todo!(),
            llvm_ir::Constant::GetElementPtr(get_element_ptr) => todo!(),
            llvm_ir::Constant::Trunc(trunc) => todo!(),
            llvm_ir::Constant::PtrToInt(ptr_to_int) => todo!(),
            llvm_ir::Constant::IntToPtr(int_to_ptr) => todo!(),
            llvm_ir::Constant::BitCast(bit_cast) => todo!(),
            llvm_ir::Constant::AddrSpaceCast(addr_space_cast) => todo!(),
            llvm_ir::Constant::PtrAuth {
                ptr,
                key,
                disc,
                addr_disc,
            } => todo!(),
        };
        Ok(self.constants.intern(const_def))
    }

    fn fold_int_binop(
        &mut self,
        lhs_ref: &ConstantRef,
        rhs_ref: &ConstantRef,
        module: &Module,
        op: impl FnOnce(i64, i64) -> i64,
    ) -> color_eyre::Result<ConstantDef> {
        let lhs_id = self.convert_const_ref(lhs_ref.clone(), module)?;
        let rhs_id = self.convert_const_ref(rhs_ref.clone(), module)?;
        let lhs = self.constants.get(lhs_id);
        let rhs = self.constants.get(rhs_id);
        match (&lhs.kind, &rhs.kind) {
            (
                ConstantKind::Scalar(ConstValue::Int(a)),
                ConstantKind::Scalar(ConstValue::Int(b)),
            ) => Ok(ConstantDef {
                ty: lhs.ty,
                kind: ConstantKind::Scalar(ConstValue::Int(op(*a, *b))),
            }),
            _ => Err(eyre!(
                "constant integer binary op requires two integer operands, got: {:?} and {:?}",
                lhs.kind,
                rhs.kind
            )),
        }
    }

    fn lower_fn_body(&mut self, func: &llvm_ir::Function) {
        let rvsdg_fn = self.get_func_by_name(&func.name).unwrap();
        // &func.basic_blocks;
        todo!("lower function basic blocks here")
    }
}

impl TypeArena {
    fn convert_type_ref(
        &mut self,
        ty: &LLVMTypeRef,
        module: &Module,
    ) -> color_eyre::Result<TypeRef> {
        Ok(match ty.as_ref() {
            llvm_ir::Type::VoidType => VOID,
            llvm_ir::Type::IntegerType { bits } => TypeRef::Scalar(int_bit_to_scalar(*bits)?),
            llvm_ir::Type::PointerType { addr_space: _ } => {
                // Opaque pointers in LLVM 17+ — no pointee type
                TypeRef::Ptr(self.intern_ptr(PtrType {
                    pointee: None,
                    alias_set: None,
                    no_escape: false,
                }))
            }
            llvm_ir::Type::FPType(fptype) => match fptype {
                llvm_ir::types::FPType::Single => TypeRef::Scalar(ScalarType::F32),
                llvm_ir::types::FPType::Double => TypeRef::Scalar(ScalarType::F64),
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
                is_packed: _,
            } => {
                let fields: Vec<StructField> = element_types
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        Ok(StructField {
                            name: None,
                            index: i as u64,
                            field_type: self.convert_type_ref(t, module)?,
                        })
                    })
                    .collect::<color_eyre::Result<_>>()?;
                TypeRef::Struct(self.intern_struct(StructDef {
                    name: None,
                    fields,
                    // calculating the struct size here requires knowing the offsets and
                    // padding which LLVM will do for us, so we only need this when not going into
                    // LLVM.
                    size: 0,
                }))
            }
            llvm_ir::Type::NamedStructType { name } => {
                match module.types.named_struct_def(name) {
                    Some(llvm_ir::types::NamedStructDef::Defined(inner_ty)) => {
                        self.convert_type_ref(inner_ty, module)?
                    }
                    Some(llvm_ir::types::NamedStructDef::Opaque) => {
                        // Opaque structs are only used behind pointers, so treat
                        // as an empty struct placeholder
                        TypeRef::Struct(self.intern_struct(StructDef {
                            name: Some(name.clone()),
                            fields: vec![],
                            size: 0,
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

impl FnDecl {
    fn from_fn_decleration(
        func: &FunctionDeclaration,
        types: &mut TypeArena,
        module: &Module,
    ) -> color_eyre::Result<Self> {
        let ret_ty = types.convert_type_ref(&func.return_type, module)?;
        let return_types = if ret_ty == VOID { vec![] } else { vec![ret_ty] };

        Ok(Self {
            name: func.name.clone(),
            params: func
                .parameters
                .iter()
                .map(|param| {
                    Ok(Param {
                        ty: types.convert_type_ref(&param.ty.clone(), module)?,
                        flags: ParamAttrFlags::empty(),
                        extra: None,
                    })
                })
                .collect::<color_eyre::Result<_>>()?,
            return_types,
            return_attrs: ParamAttrs {
                flags: ParamAttrFlags::empty(),
                extra: None,
            },
            linkage_type: convert_linkage(func.linkage),
            calling_convention: convert_calling_convention(func.calling_convention),
            is_var_arg: func.is_var_arg,
            is_exported: func.visibility != llvm_ir::module::Visibility::Hidden,
            inline_hint: InlineHint::Auto,
            visibility: convert_visibility(func.visibility),
            attrs: FnAttrs {
                flags: FnAttrFlags::empty(),
                alignment: if func.alignment > 0 {
                    Some(func.alignment)
                } else {
                    None
                },
                section: None,
            },
        })
    }

    fn from_fn(
        func: &llvm_ir::Function,
        types: &mut TypeArena,
        module: &Module,
    ) -> color_eyre::Result<Self> {
        // Function and FunctionDeclaration share the same signature fields,
        // so we construct a temporary FunctionDeclaration to reuse the logic.
        // TODO: this is very imperformant, replace with a trait that allows us to inline getting
        // each field
        let as_decl = FunctionDeclaration {
            name: func.name.clone(),
            parameters: func.parameters.clone(),
            is_var_arg: func.is_var_arg,
            return_type: func.return_type.clone(),
            return_attributes: func.return_attributes.clone(),
            linkage: func.linkage,
            visibility: func.visibility,
            dll_storage_class: func.dll_storage_class,
            calling_convention: func.calling_convention,
            alignment: func.alignment,
            garbage_collector_name: func.garbage_collector_name.clone(),
            debugloc: func.debugloc.clone(),
        };
        let mut decl = Self::from_fn_decleration(&as_decl, types, module)?;

        // Apply function-level attributes that declarations don't have
        for attr in &func.function_attributes {
            match attr {
                llvm_ir::function::FunctionAttribute::NoReturn => {
                    decl.attrs.flags |= FnAttrFlags::NO_RETURN;
                }
                llvm_ir::function::FunctionAttribute::NoUnwind => {
                    decl.attrs.flags |= FnAttrFlags::NO_UNWIND;
                }
                llvm_ir::function::FunctionAttribute::ReadOnly => {
                    decl.attrs.flags |= FnAttrFlags::READ_ONLY;
                }
                llvm_ir::function::FunctionAttribute::ReadNone => {
                    decl.attrs.flags |= FnAttrFlags::NO_MEMORY;
                }
                llvm_ir::function::FunctionAttribute::WriteOnly => {
                    decl.attrs.flags |= FnAttrFlags::WRITE_ONLY;
                }
                llvm_ir::function::FunctionAttribute::NoRecurse => {
                    decl.attrs.flags |= FnAttrFlags::NO_RECURSE;
                }
                llvm_ir::function::FunctionAttribute::NoInline => {
                    decl.attrs.flags |= FnAttrFlags::NO_INLINE;
                    decl.inline_hint = InlineHint::Never;
                }
                llvm_ir::function::FunctionAttribute::AlwaysInline => {
                    decl.attrs.flags |= FnAttrFlags::ALWAYS_INLINE;
                    decl.inline_hint = InlineHint::Always;
                }
                llvm_ir::function::FunctionAttribute::Cold => {
                    decl.attrs.flags |= FnAttrFlags::COLD;
                }
                _ => {} // ignore attributes we don't model yet
            }
        }

        if let Some(section) = &func.section {
            decl.attrs.section = Some(section.clone());
        }

        Ok(decl)
    }
}

fn convert_linkage(linkage: llvm_ir::module::Linkage) -> Linkage {
    match linkage {
        llvm_ir::module::Linkage::Private | llvm_ir::module::Linkage::Internal => Linkage::Internal,
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

fn convert_calling_convention(cc: llvm_ir::function::CallingConvention) -> CallingConvention {
    match cc {
        llvm_ir::function::CallingConvention::C => CallingConvention::C,
        llvm_ir::function::CallingConvention::Fast => CallingConvention::Fast,
        llvm_ir::function::CallingConvention::Cold => CallingConvention::Cold,
        llvm_ir::function::CallingConvention::GHC => CallingConvention::GHC,
        llvm_ir::function::CallingConvention::HiPE => CallingConvention::HiPE,
        llvm_ir::function::CallingConvention::PreserveMost => CallingConvention::PreserveMost,
        llvm_ir::function::CallingConvention::PreserveAll => CallingConvention::PreserveAll,
        llvm_ir::function::CallingConvention::Swift => CallingConvention::Swift,
        llvm_ir::function::CallingConvention::CXX_FastTLS => CallingConvention::Fast,
        other => {
            eprintln!("warning: unsupported calling convention {other:?}, defaulting to C");
            CallingConvention::C
        }
    }
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
///   - i8 `-1`:  LLVM stores 0xFF (255). Sign-extend → -1i64. Lowering: -1i64 as u64 = 0xFFFFFFFF_FFFFFFFF, truncated to i8 = 0xFF. ✓
///   - i8 `127`: LLVM stores 0x7F (127). Sign-extend → 127i64. ✓
///   - i32 `-1`: LLVM stores 0xFFFFFFFF. Sign-extend → -1i64. ✓
///   - i64 `-1`: LLVM stores 0xFFFFFFFF_FFFFFFFF. Cast → -1i64. ✓
fn sign_extend_to_i64(value: u64, bits: u32) -> i64 {
    if bits >= 64 {
        value as i64
    } else {
        let shift = 64 - bits;
        ((value as i64) << shift) >> shift
    }
}

fn int_bit_to_scalar(bits: u32) -> color_eyre::Result<ScalarType> {
    Ok(match bits {
        1 => ScalarType::Bool,
        8 => ScalarType::I8,
        16 => ScalarType::I16,
        32 => ScalarType::I32,
        64 => ScalarType::I64,
        128 => ScalarType::I128,
        _ => Err(eyre!("unsupported integer width: {bits}"))?,
    })
}
