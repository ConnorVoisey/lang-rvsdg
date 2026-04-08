use crate::rvsdg::{
    FCmpPred, ICmpPred, RVSDGMod, ValueId, ValueKind,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
    types::TypeRef,
};
use inkwell::{
    FloatPredicate, IntPredicate,
    builder::BuilderError,
    types::BasicType,
    values::{
        BasicMetadataValueEnum, BasicValueEnum, InstructionValue, ValueKind as LLVMValueKind,
    },
};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_value<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        value_id: ValueId,
    ) -> Result<Option<BasicValueEnum<'ctx>>, BuilderError> {
        if let Some(val) = mapper.get_val(value_id) {
            return Ok(Some(*val));
        }

        let value = &self.values[value_id.0 as usize];
        let lowered_val = match value.kind {
            ValueKind::Const(const_value) => {
                Some(self.lower_const_value(llvm_builder, mapper, &const_value, value.ty))
            }
            ValueKind::ConstPoolRef(const_id) => {
                Some(self.lower_const_id(llvm_builder, mapper, const_id))
            }
            ValueKind::GlobalRef(global_id) => {
                let glob = mapper
                    .get_global(global_id)
                    .expect("global should have be lowered to llvm earlier");
                Some(BasicValueEnum::PointerValue(glob.as_pointer_value()))
            }
            ValueKind::FuncAddr(func_id) => todo!(),
            ValueKind::Unary { op, operand } => {
                Some(self.lower_unary(llvm_builder, mapper, rvsdg_func, op, operand)?)
            }
            ValueKind::Binary {
                op,
                flags,
                left,
                right,
            } => {
                Some(self.lower_binary(llvm_builder, mapper, rvsdg_func, op, flags, left, right)?)
            }
            ValueKind::ICmp { pred, left, right } => {
                let lhs = self.expect_value(llvm_builder, mapper, rvsdg_func, left)?;
                let rhs = self.expect_value(llvm_builder, mapper, rvsdg_func, right)?;
                let int_pred = match pred {
                    ICmpPred::Eq => IntPredicate::EQ,
                    ICmpPred::Ne => IntPredicate::NE,
                    ICmpPred::UnsignedGt => IntPredicate::UGT,
                    ICmpPred::UnsignedGe => IntPredicate::UGE,
                    ICmpPred::UnsignedLt => IntPredicate::ULT,
                    ICmpPred::UnsignedLe => IntPredicate::ULE,
                    ICmpPred::SignedGt => IntPredicate::SGT,
                    ICmpPred::SignedGe => IntPredicate::SGE,
                    ICmpPred::SignedLt => IntPredicate::SLT,
                    ICmpPred::SignedLe => IntPredicate::SLE,
                };
                Some(BasicValueEnum::IntValue(
                    llvm_builder.builder.build_int_compare(
                        int_pred,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "icmp",
                    )?,
                ))
            }
            ValueKind::FCmp { pred, left, right } => {
                let lhs = self.expect_value(llvm_builder, mapper, rvsdg_func, left)?;
                let rhs = self.expect_value(llvm_builder, mapper, rvsdg_func, right)?;
                let float_pred = match pred {
                    FCmpPred::False => FloatPredicate::PredicateFalse,
                    FCmpPred::OrderedEq => FloatPredicate::OEQ,
                    FCmpPred::OrderedGt => FloatPredicate::OGT,
                    FCmpPred::OrderedGe => FloatPredicate::OGE,
                    FCmpPred::OrderedLt => FloatPredicate::OLT,
                    FCmpPred::OrderedLe => FloatPredicate::OLE,
                    FCmpPred::OrderedNe => FloatPredicate::ONE,
                    FCmpPred::Ordered => FloatPredicate::ORD,
                    FCmpPred::UnorderedEq => FloatPredicate::UEQ,
                    FCmpPred::UnorderedGt => FloatPredicate::UGT,
                    FCmpPred::UnorderedGe => FloatPredicate::UGE,
                    FCmpPred::UnorderedLt => FloatPredicate::ULT,
                    FCmpPred::UnorderedLe => FloatPredicate::ULE,
                    FCmpPred::UnorderedNe => FloatPredicate::UNE,
                    FCmpPred::Unordered => FloatPredicate::UNO,
                    FCmpPred::True => FloatPredicate::PredicateTrue,
                };
                Some(BasicValueEnum::IntValue(
                    llvm_builder.builder.build_float_compare(
                        float_pred,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "fcmp",
                    )?,
                ))
            }
            ValueKind::Ternary {
                condition,
                true_val,
                false_val,
            } => {
                let cond = self.expect_value(llvm_builder, mapper, rvsdg_func, condition)?;
                let then_val = self.expect_value(llvm_builder, mapper, rvsdg_func, true_val)?;
                let else_val = self.expect_value(llvm_builder, mapper, rvsdg_func, false_val)?;
                Some(llvm_builder.builder.build_select(
                    cond.into_int_value(),
                    then_val,
                    else_val,
                    "select",
                )?)
            }
            ValueKind::Cast { op, value } => todo!(),
            ValueKind::ExtractLane { vector, index } => todo!(),
            ValueKind::InsertLane {
                vector,
                index,
                value,
            } => todo!(),
            ValueKind::ShuffleLanes { left, right, mask } => todo!(),
            ValueKind::ExtractField { aggregate, indices } => {
                let mut agg = self.expect_value(llvm_builder, mapper, rvsdg_func, aggregate)?;
                let idx_slice = self.u32_pool.get(indices);
                for &idx in idx_slice {
                    agg = match agg {
                        BasicValueEnum::ArrayValue(av) => llvm_builder
                            .builder
                            .build_extract_value(av, idx, "extract")?,
                        BasicValueEnum::StructValue(sv) => llvm_builder
                            .builder
                            .build_extract_value(sv, idx, "extract")?,
                        _ => panic!("extractvalue requires an aggregate (array or struct) value"),
                    };
                }
                Some(agg)
            }
            ValueKind::InsertField {
                aggregate,
                value,
                indices,
            } => todo!(),
            ValueKind::PtrOffset {
                base,
                base_type,
                indices,
                inbounds,
            } => todo!(),
            ValueKind::Load {
                state: _,
                addr,
                loaded_type,
                align,
                volatile,
            } => {
                let ptr = self.expect_value(llvm_builder, mapper, rvsdg_func, addr)?;
                let pointee_type = self.type_to_basic_type_llvm(llvm_builder.context, loaded_type);
                let load = llvm_builder.builder.build_load(
                    pointee_type,
                    ptr.into_pointer_value(),
                    "load",
                )?;
                let inst = llvm_builder
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_last_instruction()
                    .unwrap();
                // build_load always produces an instruction-backed value
                if let Some(a) = align {
                    inst.set_alignment(a).unwrap();
                }
                if volatile {
                    inst.set_volatile(true).unwrap();
                }
                // Load is a multi-output node (state + value). Write value to Project slot.
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, load);
                None
            }
            ValueKind::Store {
                state,
                addr,
                value,
                align,
                volatile,
            } => todo!(),
            ValueKind::Alloca {
                state,
                elem_type,
                count,
            } => todo!(),
            ValueKind::AtomicLoad {
                state,
                addr,
                loaded_type,
                ordering,
                align,
            } => todo!(),
            ValueKind::AtomicStore {
                state,
                addr,
                value,
                ordering,
                align,
            } => todo!(),
            ValueKind::AtomicReadModifyWrite {
                state,
                addr,
                value,
                op,
                ordering,
            } => todo!(),
            ValueKind::CompareAndSwap {
                state,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            } => todo!(),
            ValueKind::Fence { state, ordering } => todo!(),
            ValueKind::Freeze { value } => todo!(),
            ValueKind::Intrinsic { op, state, args } => todo!(),
            ValueKind::Lambda { region, func_id } => None,
            ValueKind::Theta {
                loop_vars,
                condition,
                state: _,
                region_id: region,
            } => self.lower_theta(
                llvm_builder,
                mapper,
                rvsdg_func,
                value_id,
                loop_vars,
                condition,
                region,
            )?,
            ValueKind::Gamma {
                condition,
                inputs,
                state: _,
                regions,
            } => self.lower_gamma(
                llvm_builder,
                mapper,
                rvsdg_func,
                value_id,
                condition,
                inputs,
                regions,
            )?,
            ValueKind::Phi { region, rv_count } => todo!(),
            ValueKind::Call {
                state: _,
                fn_id,
                args,
            } => {
                let func = mapper
                    .get_fn(fn_id)
                    .expect("called function should have been registered");
                let llvm_args: Vec<BasicMetadataValueEnum<'ctx>> = self
                    .value_pool
                    .get(args)
                    .iter()
                    .map(|&arg_id| {
                        self.expect_value(llvm_builder, mapper, rvsdg_func, arg_id)
                            .expect("failed to lower call argument")
                            .into()
                    })
                    .collect();
                match llvm_builder
                    .builder
                    .build_call(func, &llvm_args, "call")
                    .expect("failed to build call")
                    .try_as_basic_value()
                {
                    LLVMValueKind::Basic(val) => Some(val),
                    LLVMValueKind::Instruction(_) => None,
                }
            }
            ValueKind::CallIndirect {
                state: _,
                callee,
                args,
            } => {
                let callee_val = self.expect_value(llvm_builder, mapper, rvsdg_func, callee)?;
                let callee_value = &self.values[callee.0 as usize];
                let func_type_id = match callee_value.ty {
                    TypeRef::Ptr(ptr_id) => {
                        let ptr_type = self.types.get_ptr(ptr_id);
                        match ptr_type.pointee {
                            Some(TypeRef::Func(id)) => id,
                            _ => {
                                panic!("indirect call callee must be a pointer to a function type")
                            }
                        }
                    }
                    TypeRef::Func(id) => id,
                    _ => {
                        panic!("indirect call callee must have a function or function-pointer type")
                    }
                };
                let func_type_def = self.types.get_fn(func_type_id);
                let ret_type =
                    self.type_to_basic_type_llvm(llvm_builder.context, func_type_def.ret);
                let param_types: Vec<_> = func_type_def
                    .params
                    .iter()
                    .map(|&ty| self.type_to_basic_meta_llvm(llvm_builder.context, ty))
                    .collect();
                let llvm_fn_type = ret_type.fn_type(&param_types, func_type_def.is_var_arg);
                let llvm_args: Vec<BasicMetadataValueEnum<'ctx>> = self
                    .value_pool
                    .get(args)
                    .iter()
                    .map(|&arg_id| {
                        self.expect_value(llvm_builder, mapper, rvsdg_func, arg_id)
                            .expect("failed to lower indirect call argument")
                            .into()
                    })
                    .collect();
                match llvm_builder
                    .builder
                    .build_indirect_call(
                        llvm_fn_type,
                        callee_val.into_pointer_value(),
                        &llvm_args,
                        "callind",
                    )
                    .expect("failed to build indirect call")
                    .try_as_basic_value()
                {
                    LLVMValueKind::Basic(val) => Some(val),
                    LLVMValueKind::Instruction(_) => None,
                }
            }
            ValueKind::Project { call, index: _ } => {
                // Ensure the parent node has been lowered.
                // Multi-output nodes (gamma, theta, call) write their results
                // directly to the Project slots in the mapper during lowering.
                // Single-output nodes return their value which we use as fallback.
                self.lower_value(llvm_builder, mapper, rvsdg_func, call)?;
                // Check if the parent populated our slot in the mapper
                if let Some(val) = mapper.get_val(value_id) {
                    Some(*val)
                } else {
                    // Fallback for single-output nodes (e.g. Call returning one value)
                    *mapper.get_val(call)
                }
            }
            ValueKind::RegionParam { index, ty } => {
                unreachable!("RegionParam should have been pre-populated in the mapper")
            }
            ValueKind::RegionResult { values, state } => None,
        };

        if let Some(val) = lowered_val {
            mapper.set_val(value_id, val);
        }
        Ok(lowered_val)
    }

    /// Lower a value that is expected to produce a result (e.g. an operand).
    /// Panics if the value does not produce an LLVM value.
    #[inline]
    pub(crate) fn expect_value<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        value_id: ValueId,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        self.lower_value(llvm_builder, mapper, rvsdg_func, value_id)
            .map(|opt| opt.expect("expected a value-producing node, got a state-only node"))
    }
}
