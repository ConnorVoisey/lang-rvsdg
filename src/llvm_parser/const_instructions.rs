use std::any::Any;

use crate::{
    llvm_parser::{global_name_string, int_bit_to_scalar, sign_extend_to_i64},
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
            llvm_ir::Constant::Struct { values, .. } => {
                // A struct constant is an aggregate of its field constants,
                // same as an array (the backend dispatches on the type:
                // arrays use const_array, structs use const_struct). The
                // struct type comes from the whole constant's type. Packing
                // is not tracked here — the backend builds non-packed
                // structs, and csmith is run with --no-packed-struct.
                let llvm_ty = module.types.type_of(const_ref.as_ref());
                let ty = self.types.convert_type_ref(&llvm_ty, module)?;
                let ids = values
                    .iter()
                    .map(|v| self.convert_const_ref(v.clone(), module))
                    .collect::<Result<Vec<_>, _>>()?;
                let span = self.constants.id_pool.push_slice(&ids);
                ConstantDef {
                    ty,
                    kind: ConstantKind::Aggregate(span),
                }
            }
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
                let name_str = global_name_string(name);
                let ty = self.types.convert_type_ref(ty, module)?;
                if let Some(&global_id) = self.global_map.get(&name_str) {
                    ConstantDef {
                        ty,
                        kind: ConstantKind::GlobalAddr(global_id),
                    }
                } else if let Some(&func_id) = self.fn_map.get(&name_str) {
                    ConstantDef {
                        ty,
                        kind: ConstantKind::FuncAddr(func_id),
                    }
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
            llvm_ir::Constant::GetElementPtr(gep) => {
                let base = self.convert_const_ref(gep.address.clone(), module)?;
                let base_type = self.constants.get(base).ty;
                // llvm-ir drops the source element type on a constant GEP
                // (opaque pointers), so recover it from the index shape:
                //   - one index  → LLVM's canonical byte form
                //     `getelementptr (i8, ptr base, i64 offset)`; index over i8.
                //   - many indices → a typed aggregate access
                //     `getelementptr (T, ptr base, 0, k, …)`; the source type
                //     T is what `base` points to.
                let source_type = if gep.indices.len() == 1 {
                    TypeRef::Scalar(ScalarType::I8)
                } else {
                    self.const_pointee_type(base).ok_or_else(|| {
                        eyre!("could not infer source type for multi-index constant getelementptr")
                    })?
                };
                let index_ids = gep
                    .indices
                    .iter()
                    .map(|i| self.convert_const_ref(i.clone(), module))
                    .collect::<Result<Vec<_>, _>>()?;
                let indices = self.constants.id_pool.push_slice(&index_ids);
                ConstantDef {
                    ty: base_type,
                    kind: ConstantKind::GetElementPointer {
                        base,
                        source_type,
                        indices,
                        in_bounds: gep.in_bounds,
                    },
                }
            }
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

    /// The element type a pointer constant points to — the *source type* for a
    /// typed GEP applied to it. A global points to its value type; a nested
    /// constant GEP points to the element its own indices land on (its source
    /// type descended by every index after the leading pointer-stride index).
    /// `None` for pointers whose pointee we don't track (e.g. function or null
    /// pointers), which can't be a typed-GEP base.
    fn const_pointee_type(&self, id: ConstId) -> Option<TypeRef> {
        match &self.constants.get(id).kind {
            ConstantKind::GlobalAddr(global_id) => Some(self.get_global(*global_id).ty),
            ConstantKind::GetElementPointer {
                source_type,
                indices,
                ..
            } => {
                let indices = self.constants.get_aggregate_elements(*indices).to_vec();
                self.descend_type(*source_type, &indices[1..])
            }
            _ => None,
        }
    }

    /// Walk an aggregate type along constant GEP indices, returning the type
    /// reached. Array indices step into the element type; struct indices select
    /// a field by its constant index. `None` on a non-aggregate or bad index.
    fn descend_type(&self, mut ty: TypeRef, indices: &[ConstId]) -> Option<TypeRef> {
        for &index in indices {
            ty = match ty {
                TypeRef::Array(array_id) => self.types.get_array(array_id).element,
                TypeRef::Struct(struct_id) => {
                    let field = self.const_int_value(index)? as usize;
                    self.types
                        .get_struct(struct_id)
                        .fields
                        .get(field)?
                        .field_type
                }
                _ => return None,
            };
        }
        Some(ty)
    }

    /// The integer value of a scalar integer constant, if it is one.
    fn const_int_value(&self, id: ConstId) -> Option<i64> {
        match self.constants.get(id).kind {
            ConstantKind::Scalar(ConstValue::Int(v)) => Some(v),
            _ => None,
        }
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
