use crate::rvsdg::{
    FCmpPred, ICmpPred, ValueId, ValueKind,
    lower_to_llvm::{FunctionLowerer, memory::ordering_to_llvm},
    types::{TypeRef, VOID},
};
use color_eyre::eyre::{bail, eyre};
use inkwell::{
    FloatPredicate, IntPredicate,
    types::BasicType,
    values::{BasicMetadataValueEnum, BasicValueEnum, ValueKind as LLVMValueKind},
};

impl<'m, 'a, 'ctx> FunctionLowerer<'m, 'a, 'ctx> {
    #[inline]
    pub(crate) fn lower_value(
        &mut self,
        value_id: ValueId,
    ) -> color_eyre::Result<Option<BasicValueEnum<'ctx>>> {
        if let Some(val) = self.get_val(value_id) {
            return Ok(Some(val));
        }

        let graph = self.graph;
        let value_kind = *graph.get_value_kind(value_id);
        let value_type = *graph.get_value_type(value_id);
        let lowered_val = match value_kind {
            ValueKind::Const(const_value) => {
                Some(self.mod_lower.lower_const_value(&const_value, value_type)?)
            }
            ValueKind::ConstPoolRef(const_id) => Some(self.mod_lower.lower_const_id(const_id)?),
            // Ordering only; emits no instruction.
            ValueKind::StateMerge { .. } => None,
            ValueKind::GlobalRef(global_id) => {
                let glob = self
                    .mod_lower
                    .get_global(global_id)
                    .ok_or_else(|| eyre!("global {global_id:?} was not lowered before use"))?;
                Some(BasicValueEnum::PointerValue(glob.as_pointer_value()))
            }
            ValueKind::FuncAddr(func_id) => {
                let func = self
                    .mod_lower
                    .get_fn(func_id)
                    .ok_or_else(|| eyre!("function {func_id:?} was not registered before use"))?;
                Some(BasicValueEnum::PointerValue(
                    func.as_global_value().as_pointer_value(),
                ))
            }
            ValueKind::Unary { op, operand } => Some(self.lower_unary(op, operand)?),
            ValueKind::Binary {
                op,
                flags,
                left,
                right,
            } => Some(self.lower_binary(op, flags, left, right)?),
            ValueKind::ICmp { pred, left, right } => {
                let lhs = self.expect_value(left)?;
                let rhs = self.expect_value(right)?;
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
                // LLVM's icmp also accepts pointer operands (`icmp eq ptr
                // %p, null`); both operands always share one type, so the
                // left one decides which build variant applies. Vector
                // compares (int or pointer lanes) are not lowered yet;
                // fail with a message instead of an inkwell unwrap panic.
                if lhs.is_vector_value() {
                    bail!("vector icmp lowering is not implemented");
                }
                let result = if lhs.is_pointer_value() {
                    self.builder.build_int_compare(
                        int_pred,
                        lhs.into_pointer_value(),
                        rhs.into_pointer_value(),
                        "icmp",
                    )?
                } else {
                    self.builder.build_int_compare(
                        int_pred,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "icmp",
                    )?
                };
                Some(BasicValueEnum::IntValue(result))
            }
            ValueKind::FCmp { pred, left, right } => {
                let lhs = self.expect_value(left)?;
                let rhs = self.expect_value(right)?;
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
                Some(BasicValueEnum::IntValue(self.builder.build_float_compare(
                    float_pred,
                    lhs.into_float_value(),
                    rhs.into_float_value(),
                    "fcmp",
                )?))
            }
            ValueKind::Ternary {
                condition,
                true_val,
                false_val,
            } => {
                let cond = self.expect_value(condition)?;
                let then_val = self.expect_value(true_val)?;
                let else_val = self.expect_value(false_val)?;
                Some(self.builder.build_select(
                    cond.into_int_value(),
                    then_val,
                    else_val,
                    "select",
                )?)
            }
            ValueKind::Cast { op, value: operand } => {
                Some(self.lower_cast(op, operand, value_type)?)
            }

            ValueKind::ExtractLane { .. }
            | ValueKind::InsertLane { .. }
            | ValueKind::ShuffleLanes { .. } => todo!("lower simd values"),

            ValueKind::ExtractField { aggregate, indices } => {
                let mut agg = self.expect_value(aggregate)?;
                let idx_slice = graph.u32_pool.get(indices);
                for &idx in idx_slice {
                    agg = match agg {
                        BasicValueEnum::ArrayValue(av) => {
                            self.builder.build_extract_value(av, idx, "extract")?
                        }
                        BasicValueEnum::StructValue(sv) => {
                            self.builder.build_extract_value(sv, idx, "extract")?
                        }
                        other => bail!(
                            "extractvalue requires an aggregate (array or struct) value, got {other:?}"
                        ),
                    };
                }
                Some(agg)
            }
            ValueKind::InsertField {
                aggregate,
                value: insert_val,
                indices,
            } => {
                let agg = self.expect_value(aggregate)?;
                let val = self.expect_value(insert_val)?;
                let idx_slice = graph.u32_pool.get(indices);

                // TODO: support multi-index insertvalue (e.g. insertvalue %s, i32 42, 0, 1)
                // by extracting nested aggregates, inserting at the leaf, and inserting
                // modified aggregates back up the chain.
                if idx_slice.len() != 1 {
                    bail!("multi-index insertvalue is not yet supported");
                }

                let result = match agg {
                    BasicValueEnum::ArrayValue(av) => {
                        self.builder
                            .build_insert_value(av, val, idx_slice[0], "insert")?
                    }
                    BasicValueEnum::StructValue(sv) => {
                        self.builder
                            .build_insert_value(sv, val, idx_slice[0], "insert")?
                    }
                    other => bail!(
                        "insertvalue requires an aggregate (array or struct) value, got {other:?}"
                    ),
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
                let ptr = self.expect_value(base)?;
                let pointee_type = self.mod_lower.type_to_basic_type_llvm(base_type)?;
                let idx_vals: Vec<_> = graph
                    .value_pool
                    .get(indices)
                    .iter()
                    .map(|&id| self.expect_value(id).map(|v| v.into_int_value()))
                    .collect::<color_eyre::Result<_>>()?;
                // SAFETY: inkwell marks every GEP constructor `unsafe`
                // because mismatched indices/type are UB in LLVM.
                // `pointee_type` and the index list both come from the
                // same PtrOffset node, whose shape the frontend
                // validated against the source element type at
                // conversion (it refuses unrecoverable shapes). Same
                // contract as the constant GEP lowering in const_val.rs.
                let result = unsafe {
                    if inbounds {
                        self.builder.build_in_bounds_gep(
                            pointee_type,
                            ptr.into_pointer_value(),
                            &idx_vals,
                            "gep",
                        )?
                    } else {
                        self.builder.build_gep(
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
                self.lower_load(addr, loaded_type, align, volatile, value_id)?;
                None
            }
            ValueKind::Store {
                state: _,
                addr,
                value,
                align,
                volatile,
            } => {
                self.lower_store(addr, value, align, volatile)?;

                None
            }
            ValueKind::Alloca {
                state: _,
                elem_type,
                count,
                align,
            } => {
                self.lower_alloca(value_id, elem_type, count, align)?;
                None
            }

            ValueKind::AtomicLoad {
                state: _,
                addr,
                loaded_type,
                ordering,
                align,
                volatile,
            } => {
                self.lower_atomic_load(addr, loaded_type, ordering, align, volatile, value_id)?;
                None
            }
            ValueKind::AtomicStore {
                state: _,
                addr,
                value,
                ordering,
                align,
                volatile,
            } => {
                self.lower_atomic_store(addr, value, ordering, align, volatile)?;
                None
            }
            ValueKind::AtomicReadModifyWrite {
                state: _,
                addr,
                value,
                op,
                ordering,
                volatile,
            } => {
                self.lower_atomic_read_modify_write(value_id, addr, value, op, ordering, volatile)?;
                None
            }

            ValueKind::CompareAndSwap {
                state: _,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
                volatile,
            } => {
                self.lower_compare_and_swap(
                    value_id,
                    addr,
                    expected,
                    desired,
                    success_ordering,
                    failure_ordering,
                    volatile,
                )?;
                None
            }
            ValueKind::Fence { state: _, ordering } => {
                // The second argument is LLVM's single-thread flag; 0 is the
                // default cross-thread (system) synchronisation scope. The
                // frontend drops the input's scope (see
                // convert_mem_ordering), so singlethread fences are
                // strengthened to system scope here -- correct, but emits
                // real hardware fences where none were needed. The name
                // must be empty: a fence is void and LLVM rejects named
                // void values. Not single-thread: syncscope is dropped at
                // parse (system scope), so the fence is cross-thread.
                self.builder
                    .build_fence(ordering_to_llvm(ordering), false, "")?;
                None
            }
            ValueKind::Freeze { .. } => todo!(),
            ValueKind::Match {
                input,
                arms,
                default,
                alternatives: _,
            } => {
                // Lower a control/predicate value to its `i32` alternative
                // index via a select chain: each arm contributes
                // `input == case ? alt : <rest>`, folded from the default
                // outward so the first matching arm wins.
                let input_val = self.expect_value(input)?.into_int_value();
                let input_ty = input_val.get_type();
                let out_ty = self.mod_lower.context.i32_type();
                let arm_slice = graph.match_arm_pool.get(arms);
                let mut acc = out_ty.const_int(default as u64, false);
                for arm in arm_slice.iter().rev() {
                    let case = input_ty.const_int(arm.value as u64, true);
                    let is_match = self.builder.build_int_compare(
                        IntPredicate::EQ,
                        input_val,
                        case,
                        "match.cmp",
                    )?;
                    let alt = out_ty.const_int(arm.alternative as u64, false);
                    acc = self
                        .builder
                        .build_select(is_match, alt, acc, "match.sel")?
                        .into_int_value();
                }
                Some(BasicValueEnum::IntValue(acc))
            }
            ValueKind::Intrinsic { op, state: _, args } => {
                self.lower_intrinsic(op, args, value_id)?;
                None
            }
            ValueKind::Theta {
                loop_vars,
                condition,
                region_id: region,
            } => {
                self.begin_control(value_id)?;
                let lowered = self.lower_theta(value_id, loop_vars, condition, region)?;
                self.finish_control(value_id);
                lowered
            }
            ValueKind::Gamma {
                condition,
                inputs,
                regions,
            } => {
                self.begin_control(value_id)?;
                let lowered = self.lower_gamma(value_id, condition, inputs, regions)?;
                self.finish_control(value_id);
                lowered
            }
            ValueKind::Call {
                state: _,
                io_state: _,
                fn_id,
                sig,
                args,
            } => {
                let func = self
                    .mod_lower
                    .get_fn(fn_id)
                    .ok_or_else(|| eyre!("called function {fn_id:?} was not registered"))?;
                let llvm_args: Vec<BasicMetadataValueEnum<'ctx>> = graph
                    .value_pool
                    .get(args)
                    .iter()
                    .map(|&arg_id| self.expect_value(arg_id).map(|v| v.into()))
                    .collect::<color_eyre::Result<_>>()?;
                let call_site = self.builder.build_call(func, &llvm_args, "call")?;
                let tables = self.mod_lower.tables;
                self.mod_lower
                    .apply_call_site_abi(call_site, tables.signatures.get(sig))?;
                match call_site.try_as_basic_value() {
                    LLVMValueKind::Basic(val) => Some(val),
                    LLVMValueKind::Instruction(_) => None,
                }
            }
            ValueKind::CallIndirect {
                state: _,
                io_state: _,
                callee,
                sig,
                args,
            } => {
                let callee_val = self.expect_value(callee)?;
                let tables = self.mod_lower.tables;
                let signature = tables.signatures.get(sig);
                let func_type_def = tables.types.get_fn(signature.func_type);
                let param_types: Vec<_> = func_type_def
                    .params
                    .iter()
                    .map(|&ty| self.mod_lower.type_to_basic_meta_llvm(ty))
                    .collect::<color_eyre::Result<_>>()?;
                // A void return is not a BasicType in LLVM, so build the
                // function type through `void_type()` in that case (mirrors
                // `register_fn`). Common for `noreturn` callees like abort().
                let llvm_fn_type = if func_type_def.ret == VOID {
                    self.mod_lower
                        .context
                        .void_type()
                        .fn_type(&param_types, func_type_def.is_var_arg)
                } else {
                    self.mod_lower
                        .type_to_basic_type_llvm(func_type_def.ret)?
                        .fn_type(&param_types, func_type_def.is_var_arg)
                };
                let llvm_args: Vec<BasicMetadataValueEnum<'ctx>> = graph
                    .value_pool
                    .get(args)
                    .iter()
                    .map(|&arg_id| self.expect_value(arg_id).map(|v| v.into()))
                    .collect::<color_eyre::Result<_>>()?;
                let call_site = self.builder.build_indirect_call(
                    llvm_fn_type,
                    callee_val.into_pointer_value(),
                    &llvm_args,
                    "callind",
                )?;
                self.mod_lower.apply_call_site_abi(call_site, signature)?;
                match call_site.try_as_basic_value() {
                    LLVMValueKind::Basic(val) => Some(val),
                    LLVMValueKind::Instruction(_) => None,
                }
            }
            // A construct's state projections order chains only: no
            // instruction, and no demand on the construct either --
            // positional lowering already emitted it, and multi-output
            // nodes are not memoised as single values, so touching the
            // construct here would lower it a second time.
            ValueKind::Project { .. } if matches!(value_type, TypeRef::State(_)) => None,
            ValueKind::Project { call, index: _ } => {
                // Ensure the parent node has been lowered.
                // Multi-output nodes (gamma, theta, call) write their results
                // directly to the Project slots in the lowerer during lowering.
                // Single-output nodes return their value which we use as fallback.
                self.lower_value(call)?;
                // Check if the parent populated our slot in the lowerer
                if let Some(val) = self.get_val(value_id) {
                    Some(val)
                } else {
                    // Fallback for single-output nodes (e.g. Call returning one value)
                    self.get_val(call)
                }
            }
            ValueKind::RegionParam { .. } => {
                bail!("RegionParam {value_id:?} was not pre-populated in the lowerer")
            }
        };

        if let Some(val) = lowered_val {
            self.set_val(value_id, val);
        }
        Ok(lowered_val)
    }

    /// Lower a value that is expected to produce a result (e.g. an operand).
    /// Panics if the value does not produce an LLVM value.
    #[inline]
    pub(crate) fn expect_value(
        &mut self,
        value_id: ValueId,
    ) -> color_eyre::Result<BasicValueEnum<'ctx>> {
        self.lower_value(value_id)?
            .ok_or_else(|| eyre!("expected a value-producing node, got a state-only node"))
    }

    /// Fetch one of a region's RESULT values after its body has been
    /// lowered. Body values come straight out of the lowerer (the region
    /// walk lowered them); region-free values (interned constants and
    /// symbol references, which belong to no region and are never
    /// walked) are materialised on demand -- LLVM constants need no
    /// instructions, so this is safe at any builder position.
    pub(crate) fn lowered_result(
        &mut self,
        result_id: ValueId,
    ) -> color_eyre::Result<Option<BasicValueEnum<'ctx>>> {
        if self.graph.get_value_kind(result_id).is_region_free() {
            return self.lower_value(result_id);
        }
        Ok(self.get_val(result_id))
    }
}
