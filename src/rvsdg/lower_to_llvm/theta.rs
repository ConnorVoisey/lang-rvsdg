use crate::rvsdg::{
    RVSDGMod, RegionId, ValueId, ValuesSpan,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
};
use inkwell::{
    builder::BuilderError,
    values::{BasicValue, BasicValueEnum},
};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_theta<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        theta_id: ValueId,
        loop_vars: ValuesSpan,
        condition: ValueId,
        region_id: RegionId,
    ) -> Result<Option<BasicValueEnum<'ctx>>, BuilderError> {
        let input_ids = self.value_pool.get(loop_vars).to_vec();
        // This lookup is probably isn't required, I could just pass in the function instead,
        // but this requires keeping track of it
        let func = mapper
            .get_fn(rvsdg_func.id)
            .expect("function should have been lowered before regions are created");

        // Create all basic blocks upfront
        let entry_bb = llvm_builder.builder.get_insert_block().unwrap();
        let loop_bb = llvm_builder.context.append_basic_block(func, "theta.loop");
        let exit_bb = llvm_builder.context.append_basic_block(func, "theta.exit");

        llvm_builder.builder.build_unconditional_branch(loop_bb)?;
        llvm_builder.builder.position_at_end(loop_bb);

        // Build phis at top of loop_bb, add initial incoming from entry
        let initial_values = input_ids.iter().map(|&id| {
            self.expect_value(llvm_builder, mapper, rvsdg_func, id)
                // TODO: replace the unwrap with correct error handling
                .unwrap()
        });

        let phis: Vec<_> = initial_values
            .map(|val| {
                let phi = llvm_builder
                    .builder
                    .build_phi(val.get_type(), "theta.phi")
                    .unwrap();
                phi.add_incoming(&[(&val as &dyn BasicValue, entry_bb)]);
                phi
            })
            .collect();

        // Bind region params to phi outputs (not initial values)
        let region = self.get_region(region_id);
        for (j, phi) in phis.iter().enumerate() {
            let param_id = ValueId(region.params.start + j as u32);
            mapper.set_val(param_id, phi.as_basic_value());
        }

        // Lower region, get results
        self.lower_region(llvm_builder, mapper, rvsdg_func, region)?;
        let result_ids = self.value_pool.get(region.results).to_vec();
        let results: Vec<BasicValueEnum> = result_ids
            .iter()
            .filter_map(|&rid| *mapper.get_val(rid))
            .collect();

        // Condition -> branch back to loop or exit
        let cond = self.expect_value(llvm_builder, mapper, rvsdg_func, condition)?;
        let actual_bb = llvm_builder.builder.get_insert_block().unwrap();
        llvm_builder
            .builder
            .build_conditional_branch(cond.into_int_value(), loop_bb, exit_bb)?;

        // Add back-edge incoming to phis
        for (i, (phi, result)) in phis.iter().zip(results.iter()).enumerate() {
            phi.add_incoming(&[(result as &dyn BasicValue, actual_bb)]);
            let project_id = ValueId(theta_id.0 + 1 + i as u32);
            mapper.set_val(project_id, *result);
        }
        llvm_builder.builder.position_at_end(exit_bb);

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, Linkage, RVSDGMod,
        builder::{BranchResult, LoopResult},
        func::FnResult,
        lower_to_llvm::test_utils::test_utils::jit_run_i32,
        types::{BOOL, I32},
        value::ConstValue,
    };

    #[test]
    fn theta_simple() {
        // let x = 0;
        // do { x = x + 1 } while(x < 10)
        // => 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(0);
            let res = rb.theta(state, &[x], |rb| {
                let loop_x = rb.param(0);
                let one = rb.const_i32(1);
                let next_x = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, one, I32);
                let ten = rb.const_i32(10);
                let condition = rb.icmp(ICmpPred::SignedLt, next_x, ten);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn theta_single_iteration() {
        // do { x = x + 5 } while(false)
        // condition is always false, so body executes exactly once
        // => 5
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(0);
            let res = rb.theta(state, &[x], |rb| {
                let loop_x = rb.param(0);
                let five = rb.const_i32(5);
                let next_x = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, five, I32);
                let condition = rb.constant(BOOL, ConstValue::Int(0));
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 5);
    }

    #[test]
    fn theta_multiple_loop_vars() {
        // x = 0, y = 100
        // do { x = x + 1; y = y - 10 } while(x < 5)
        // => x = 5, y = 50
        // returns x * y = 250
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(0);
            let y = rb.const_i32(100);
            let res = rb.theta(state, &[x, y], |rb| {
                let loop_x = rb.param(0);
                let loop_y = rb.param(1);
                let one = rb.const_i32(1);
                let ten = rb.const_i32(10);
                let next_x = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, one, I32);
                let next_y = rb.binary(BinaryOp::Sub, ArithFlags::default(), loop_y, ten, I32);
                let five = rb.const_i32(5);
                let condition = rb.icmp(ICmpPred::SignedLt, next_x, five);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x, next_y],
                }
            });
            // x * y = 5 * 50 = 250
            let result_x = res.result(0);
            let result_y = res.result(1);
            let product = rb.binary(
                BinaryOp::Mul,
                ArithFlags::default(),
                result_x,
                result_y,
                I32,
            );
            FnResult {
                state: res.state,
                values: vec![product],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 250);
    }

    #[test]
    fn theta_countdown() {
        // x = 10
        // do { x = x - 1 } while(x > 0)
        // => 0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(10);
            let res = rb.theta(state, &[x], |rb| {
                let loop_x = rb.param(0);
                let one = rb.const_i32(1);
                let next_x = rb.binary(BinaryOp::Sub, ArithFlags::default(), loop_x, one, I32);
                let zero = rb.const_i32(0);
                let condition = rb.icmp(ICmpPred::SignedGt, next_x, zero);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    #[test]
    fn theta_accumulator() {
        // sum = 0, i = 1
        // do { sum = sum + i; i = i + 1 } while(i <= 10)
        // => sum = 1+2+...+10 = 55
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let sum = rb.const_i32(0);
            let i = rb.const_i32(1);
            let res = rb.theta(state, &[sum, i], |rb| {
                let loop_sum = rb.param(0);
                let loop_i = rb.param(1);
                let next_sum =
                    rb.binary(BinaryOp::Add, ArithFlags::default(), loop_sum, loop_i, I32);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let ten = rb.const_i32(10);
                let condition = rb.icmp(ICmpPred::SignedLe, next_i, ten);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_sum, next_i],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 55);
    }

    #[test]
    fn theta_with_multiplication_in_body() {
        // Compute 2^10 = 1024 via repeated doubling
        // x = 1, i = 0
        // do { x = x * 2; i = i + 1 } while(i < 10)
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(1);
            let i = rb.const_i32(0);
            let res = rb.theta(state, &[x, i], |rb| {
                let loop_x = rb.param(0);
                let loop_i = rb.param(1);
                let two = rb.const_i32(2);
                let next_x = rb.binary(BinaryOp::Mul, ArithFlags::default(), loop_x, two, I32);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let ten = rb.const_i32(10);
                let condition = rb.icmp(ICmpPred::SignedLt, next_i, ten);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x, next_i],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 1024);
    }

    #[test]
    fn theta_nested_in_gamma() {
        // if true { do { x++ } while(x < 7) } else { 99 }
        // => 7
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(1));
            let res = rb.gamma(
                cond,
                state,
                &[],
                |rb| {
                    let x = rb.const_i32(0);
                    let loop_res = rb.theta(state, &[x], |rb| {
                        let loop_x = rb.param(0);
                        let one = rb.const_i32(1);
                        let next_x =
                            rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, one, I32);
                        let seven = rb.const_i32(7);
                        let condition = rb.icmp(ICmpPred::SignedLt, next_x, seven);
                        LoopResult {
                            condition,
                            next_state: state,
                            next_vars: vec![next_x],
                        }
                    });
                    BranchResult {
                        state,
                        values: vec![loop_res.result(0)],
                    }
                },
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(99)],
                },
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 7);
    }

    #[test]
    fn theta_gamma_inside_loop() {
        // x = 0, sum = 0
        // do {
        //     x = x + 1
        //     sum = sum + if x % 2 == 0 { x } else { 0 }
        // } while(x < 6)
        // sum of even numbers 2 + 4 + 6 = 12
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(0);
            let sum = rb.const_i32(0);
            let res = rb.theta(state, &[x, sum], |rb| {
                let loop_x = rb.param(0);
                let loop_sum = rb.param(1);
                let one = rb.const_i32(1);
                let next_x = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, one, I32);

                // if next_x % 2 == 0 { next_x } else { 0 }
                let two = rb.const_i32(2);
                let rem = rb.binary(
                    BinaryOp::UnsignedRem,
                    ArithFlags::default(),
                    next_x,
                    two,
                    I32,
                );
                let zero = rb.const_i32(0);
                let is_even = rb.icmp(ICmpPred::Eq, rem, zero);
                let branch_res = rb.gamma(
                    is_even,
                    state,
                    &[next_x],
                    |rb| BranchResult {
                        state,
                        values: vec![rb.param(0)],
                    },
                    |rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(0)],
                    },
                );
                let to_add = branch_res.result(0);
                let next_sum =
                    rb.binary(BinaryOp::Add, ArithFlags::default(), loop_sum, to_add, I32);

                let six = rb.const_i32(6);
                let condition = rb.icmp(ICmpPred::SignedLt, next_x, six);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x, next_sum],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(1)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 12);
    }

    #[test]
    fn while_loop_via_gamma_theta() {
        // while (x < 5) { x = x + 1 }
        // Equivalent to: if (x < 5) { do { x = x + 1 } while (x < 5) } else { x }
        // with x = 0 => 5
        // with x = 10 => 10 (loop body never executes)
        let build_while_loop = |init_x: i32| -> RVSDGMod {
            let mut rvsdg = RVSDGMod::new_host(String::from("test"));
            let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
            rvsdg.define_fn(func_id, |rb, state| {
                let x = rb.const_i32(init_x);
                let five = rb.const_i32(5);
                let enter_loop = rb.icmp(ICmpPred::SignedLt, x, five);
                let res = rb.gamma(
                    enter_loop,
                    state,
                    &[x],
                    |rb| {
                        // true branch: do { x++ } while (x < 5)
                        let init = rb.param(0);
                        let loop_res = rb.theta(state, &[init], |rb| {
                            let loop_x = rb.param(0);
                            let one = rb.const_i32(1);
                            let next_x =
                                rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, one, I32);
                            let five = rb.const_i32(5);
                            let condition = rb.icmp(ICmpPred::SignedLt, next_x, five);
                            LoopResult {
                                condition,
                                next_state: state,
                                next_vars: vec![next_x],
                            }
                        });
                        BranchResult {
                            state,
                            values: vec![loop_res.result(0)],
                        }
                    },
                    |rb| {
                        // false branch: condition already false, pass through
                        BranchResult {
                            state,
                            values: vec![rb.param(0)],
                        }
                    },
                );
                FnResult {
                    state: res.state,
                    values: vec![res.result(0)],
                }
            });
            rvsdg
        };

        // x starts at 0, loop runs 5 times
        assert_eq!(jit_run_i32(&build_while_loop(0), "test"), 5);
        // x starts at 10, loop never enters
        assert_eq!(jit_run_i32(&build_while_loop(10), "test"), 10);
    }
}
