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
    values::{BasicMetadataValueEnum, BasicValueEnum, ValueKind as LLVMValueKind},
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
            ValueKind::FuncAddr(func_id) => {
                let func = mapper
                    .get_fn(func_id)
                    .expect("function should have been registered");
                Some(BasicValueEnum::PointerValue(
                    func.as_global_value().as_pointer_value(),
                ))
            }
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
            ValueKind::Cast { op, value: operand } => {
                Some(self.lower_cast(llvm_builder, mapper, rvsdg_func, op, operand, value)?)
            }

            ValueKind::ExtractLane { .. }
            | ValueKind::InsertLane { .. }
            | ValueKind::ShuffleLanes { .. } => todo!("lower simd values"),

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
                value: insert_val,
                indices,
            } => {
                let agg = self.expect_value(llvm_builder, mapper, rvsdg_func, aggregate)?;
                let val = self.expect_value(llvm_builder, mapper, rvsdg_func, insert_val)?;
                let idx_slice = self.u32_pool.get(indices);
                let b = &llvm_builder.builder;

                // TODO: support multi-index insertvalue (e.g. insertvalue %s, i32 42, 0, 1)
                // by extracting nested aggregates, inserting at the leaf, and inserting
                // modified aggregates back up the chain.
                assert_eq!(
                    idx_slice.len(),
                    1,
                    "multi-index insertvalue not yet supported"
                );

                let result = match agg {
                    BasicValueEnum::ArrayValue(av) => {
                        b.build_insert_value(av, val, idx_slice[0], "insert")?
                    }
                    BasicValueEnum::StructValue(sv) => {
                        b.build_insert_value(sv, val, idx_slice[0], "insert")?
                    }
                    _ => panic!("insertvalue requires an aggregate (array or struct) value"),
                };
                // AggregateValueEnum -> BasicValueEnum
                Some(match result {
                    inkwell::values::AggregateValueEnum::ArrayValue(av) => {
                        BasicValueEnum::ArrayValue(av)
                    }
                    inkwell::values::AggregateValueEnum::StructValue(sv) => {
                        BasicValueEnum::StructValue(sv)
                    }
                })
            }
            ValueKind::PtrOffset {
                base,
                base_type,
                indices,
                inbounds,
            } => {
                let ptr = self.expect_value(llvm_builder, mapper, rvsdg_func, base)?;
                let pointee_type = self.type_to_basic_type_llvm(llvm_builder.context, base_type);
                let idx_ids = self.value_pool.get(indices).to_vec();
                let idx_vals: Vec<_> = idx_ids
                    .iter()
                    .map(|&id| {
                        self.expect_value(llvm_builder, mapper, rvsdg_func, id)
                            .expect("failed to lower GEP index")
                            .into_int_value()
                    })
                    .collect();
                let result = if inbounds {
                    unsafe {
                        llvm_builder.builder.build_in_bounds_gep(
                            pointee_type,
                            ptr.into_pointer_value(),
                            &idx_vals,
                            "gep",
                        )?
                    }
                } else {
                    unsafe {
                        llvm_builder.builder.build_gep(
                            pointee_type,
                            ptr.into_pointer_value(),
                            &idx_vals,
                            "gep",
                        )?
                    }
                };
                Some(BasicValueEnum::PointerValue(result))
            }
            ValueKind::Load {
                state: _,
                addr,
                loaded_type,
                align,
                volatile,
            } => {
                self.lower_load(
                    llvm_builder,
                    mapper,
                    rvsdg_func,
                    addr,
                    loaded_type,
                    align,
                    volatile,
                    value_id,
                )?;
                None
            }
            ValueKind::Store {
                state: _,
                addr,
                value,
                align,
                volatile,
            } => {
                self.lower_store(
                    llvm_builder,
                    mapper,
                    rvsdg_func,
                    addr,
                    value,
                    align,
                    volatile,
                )?;

                None
            }
            ValueKind::Alloca {
                state: _,
                elem_type,
                count,
            } => {
                self.lower_alloca(llvm_builder, mapper, rvsdg_func, value_id, elem_type, count)?;
                None
            }

            ValueKind::AtomicLoad { .. }
            | ValueKind::AtomicStore { .. }
            | ValueKind::AtomicReadModifyWrite { .. } => todo!("lower atomics"),

            ValueKind::CompareAndSwap {
                state: _,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            } => {
                self.lower_cmpxchg(
                    llvm_builder,
                    mapper,
                    rvsdg_func,
                    value_id,
                    addr,
                    expected,
                    desired,
                    success_ordering,
                    failure_ordering,
                )?;
                None
            }
            ValueKind::Fence { state: _, ordering } => todo!(),
            ValueKind::Freeze { value } => todo!(),
            ValueKind::Intrinsic { op, state: _, args } => {
                self.lower_intrinsic(llvm_builder, mapper, rvsdg_func, op, args, value_id)?;
                None
            }
            ValueKind::Lambda {
                region: _,
                func_id: _,
            } => None,
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
            ValueKind::RegionParam { index: _, ty: _ } => {
                unreachable!("RegionParam should have been pre-populated in the mapper")
            }
            ValueKind::RegionResult {
                values: _,
                state: _,
            } => None,
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
