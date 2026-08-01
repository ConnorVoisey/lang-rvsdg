use crate::{
    llvm_parser::{global_name_string, int_bit_to_scalar, sign_extend_to_i64},
    rvsdg::{
        ConstId, ConstValue, ConstantDef, ConstantKind,
        module_tables::ModuleTables,
        ops::CastOp,
        types::{ArrayType, PtrType, ScalarType, TypeRef, VOID},
    },
};
use color_eyre::eyre::eyre;
use llvm_ir::{ConstantRef, Module};

impl ModuleTables {
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
                // is not tracked here -- the backend builds non-packed
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
            // The address of a basic block (GNU computed goto). Not yet
            // implemented on our side: the lowering is mechanical -- assign
            // each referenced block a small integer, rewrite blockaddress
            // as inttoptr of it and indirectbr as a switch over the
            // destination list (LLVM's own IndirectBrExpand shape) -- but
            // not worth building until enough inputs need it.
            llvm_ir::Constant::BlockAddress => {
                return Err(eyre!(
                    "blockaddress constants (computed goto) are not implemented yet"
                ));
            }
            llvm_ir::Constant::GlobalReference { name, ty: _ } => {
                let name_str = global_name_string(name);
                // The VALUE of a global reference is always a pointer.
                // llvm-ir's `ty` field is the referent's type (the pointee
                // for a global, the signature for a function), which is not
                // a value type -- stamping it on the constant leaks it into
                // value positions (e.g. a function pointer crossing a loop
                // boundary types the slot as a function). The referent type
                // lives in the globals/functions tables; typed-GEP recovery
                // reads it from there (const_pointee_type).
                let ty = TypeRef::Ptr(self.types.intern_ptr(PtrType {
                    pointee: None,
                    alias_set: None,
                    no_escape: false,
                }));
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
                // The source element type comes from the constant
                // expression itself (our llvm-ir fork reads it via
                // LLVMGetGEPSourceElementType, exactly as it already did
                // for GEP instructions). No recovery heuristics: with
                // opaque pointers the base's pointee type is NOT reliable
                // -- clang's constant folder emits single-index GEPs typed
                // over element types (`&arr[2] + 1`) and field-typed
                // folded accesses, which are unrecoverable from the shape
                // alone and used to force a refusal here.
                let source_type = self
                    .types
                    .convert_type_ref(&gep.source_element_type, module)?;
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
            // Constant-expression casts. Like the int binops above these are
            // not folded to a primitive here -- pointer-valued casts (inttoptr,
            // a ptrtoint of a global address) have no scalar form -- so the cast
            // is recorded and resolved by LLVM's constant operations at lowering.
            // addrspacecast lowers as a bitcast, matching the runtime cast path.
            llvm_ir::Constant::Trunc(c) => {
                self.convert_const_cast(CastOp::Truncate, &c.operand, &c.to_type, module)?
            }
            llvm_ir::Constant::PtrToInt(c) => {
                self.convert_const_cast(CastOp::PtrToInt, &c.operand, &c.to_type, module)?
            }
            llvm_ir::Constant::IntToPtr(c) => {
                self.convert_const_cast(CastOp::IntToPtr, &c.operand, &c.to_type, module)?
            }
            llvm_ir::Constant::BitCast(c) => {
                self.convert_const_cast(CastOp::Bitcast, &c.operand, &c.to_type, module)?
            }
            llvm_ir::Constant::AddrSpaceCast(c) => {
                self.convert_const_cast(CastOp::Bitcast, &c.operand, &c.to_type, module)?
            }
            llvm_ir::Constant::PtrAuth {
                ptr: _,
                key: _,
                disc: _,
                addr_disc: _,
            } => todo!(),
        };
        Ok(self.constants.intern(const_def))
    }

    /// Build a [`ConstantKind::Cast`]: recover the operand constant and the
    /// result type (`to_type`), leaving the actual cast to LLVM's constant
    /// operations at lowering. Used for every constant-expression cast.
    fn convert_const_cast(
        &mut self,
        op: CastOp,
        operand: &ConstantRef,
        to_type: &llvm_ir::TypeRef,
        module: &Module,
    ) -> color_eyre::Result<ConstantDef> {
        let operand = self.convert_const_ref(operand.clone(), module)?;
        let ty = self.types.convert_type_ref(to_type, module)?;
        Ok(ConstantDef {
            ty,
            kind: ConstantKind::Cast { op, operand },
        })
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
