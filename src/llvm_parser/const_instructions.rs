use crate::{
    llvm_parser::{int_bit_to_scalar, sign_extend_to_i64},
    rvsdg::{
        ConstId, ConstValue, ConstantDef, ConstantKind, RVSDGMod,
        types::{ArrayType, PtrType, ScalarType, TypeRef, VOID},
    },
};
use color_eyre::eyre::eyre;
use llvm_ir::{ConstantRef, Module};

impl RVSDGMod {
    pub(super) fn convert_const_ref(
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
                name: _,
                values: _,
                is_packed: _,
            } => todo!(),
            llvm_ir::Constant::Array {
                element_type,
                elements,
            } => {
                let element = self.types.convert_type_ref(element_type, module)?;
                let array_ty_id = self.types.intern_array(ArrayType {
                    element,
                    len: elements.len() as u64,
                });
                let ids = elements
                    .iter()
                    .map(|el| self.convert_const_ref(el.clone(), module))
                    .collect::<Result<Vec<_>, _>>()?;

                let const_id_span = self.constants.id_pool.push_slice(&ids);
                ConstantDef {
                    ty: TypeRef::Array(array_ty_id),
                    kind: ConstantKind::Aggregate(const_id_span),
                }
            }
            llvm_ir::Constant::Vector(_constant_refs) => todo!(),
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
            // It has no runtime value, treat as a zero-sized placeholder.
            llvm_ir::Constant::TokenNone => ConstantDef {
                ty: VOID,
                kind: ConstantKind::Zero,
            },

            // The RVSDG IR layer does not have a concept of constant expressions,
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
            llvm_ir::Constant::ExtractElement(_extract_element) => todo!(),
            llvm_ir::Constant::InsertElement(_insert_element) => todo!(),
            llvm_ir::Constant::ShuffleVector(_shuffle_vector) => todo!(),
            llvm_ir::Constant::GetElementPtr(_get_element_ptr) => todo!(),
            llvm_ir::Constant::Trunc(_trunc) => todo!(),
            llvm_ir::Constant::PtrToInt(_ptr_to_int) => todo!(),
            llvm_ir::Constant::IntToPtr(_int_to_ptr) => todo!(),
            llvm_ir::Constant::BitCast(_bit_cast) => todo!(),
            llvm_ir::Constant::AddrSpaceCast(_addr_space_cast) => todo!(),
            llvm_ir::Constant::PtrAuth {
                ptr: _,
                key: _,
                disc: _,
                addr_disc: _,
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
}
