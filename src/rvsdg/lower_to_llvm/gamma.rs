use crate::rvsdg::{
    RVSDGMod, RegionsSpan, ValueId, ValuesSpan,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
};
use inkwell::{
    basic_block::BasicBlock,
    builder::BuilderError,
    values::{BasicValue, BasicValueEnum},
};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_gamma<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        gamma_value_id: ValueId,
        condition: ValueId,
        inputs: ValuesSpan,
        regions: RegionsSpan,
    ) -> Result<Option<BasicValueEnum<'ctx>>, BuilderError> {
        let cond = self.expect_value(llvm_builder, mapper, rvsdg_func, condition)?;
        let region_ids = self.region_pool.get(regions).to_vec();
        let input_ids = self.value_pool.get(inputs).to_vec();
        let func = mapper
            .get_fn(rvsdg_func.id)
            .expect("function should have been lowered before regions are created");

        // Create all basic blocks upfront
        let merge_bb = llvm_builder.context.append_basic_block(func, "gamma.merge");
        let region_bbs: Vec<_> = region_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                llvm_builder
                    .context
                    .append_basic_block(func, &format!("gamma.{i}"))
            })
            .collect();

        // Emit the branch from the current block
        if region_bbs.len() == 2 {
            llvm_builder.builder.build_conditional_branch(
                cond.into_int_value(),
                region_bbs[0],
                region_bbs[1],
            )?;
        } else {
            // the first block is skipped from the cases and is placed as the default
            let cases: Vec<_> = region_bbs
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &bb)| {
                    (
                        cond.get_type().into_int_type().const_int(i as u64, false),
                        bb,
                    )
                })
                .collect();
            llvm_builder
                .builder
                .build_switch(cond.into_int_value(), region_bbs[0], &cases)?;
        }

        // Lower each region, collecting (result_values, basic_block) per region
        // This is a vec of vecs and will be slow, consider replacing this with something more
        // efficient in the future
        let mut region_results: Vec<(Vec<BasicValueEnum<'ctx>>, BasicBlock<'ctx>)> =
            Vec::with_capacity(region_ids.len());
        for (i, &region_id) in region_ids.iter().enumerate() {
            let bb = region_bbs[i];
            llvm_builder.builder.position_at_end(bb);

            let region = self.get_region(region_id);

            // Bind gamma inputs to this region's params
            for j in 0..region.params.len as u32 {
                let param_id = ValueId(region.params.start + j);
                if (j as usize) < input_ids.len() {
                    let input_val =
                        self.expect_value(llvm_builder, mapper, rvsdg_func, input_ids[j as usize])?;
                    mapper.set_val(param_id, input_val);
                }
            }

            self.lower_region(llvm_builder, mapper, rvsdg_func, region)?;

            // Collect result values from the region
            let result_ids = self.value_pool.get(region.results).to_vec();
            let results: Vec<BasicValueEnum<'ctx>> = result_ids
                .iter()
                .filter_map(|&rid| *mapper.get_val(rid))
                .collect();

            // lowering the region could insert a new basic block, so get the current basic block
            // here
            let actual_bb = llvm_builder.builder.get_insert_block().unwrap();
            llvm_builder.builder.build_unconditional_branch(merge_bb)?;
            region_results.push((results, actual_bb));
        }

        // Build phi nodes in the merge block and write results to Project slots
        llvm_builder.builder.position_at_end(merge_bb);
        let num_results = region_results.first().map(|(r, _)| r.len()).unwrap_or(0);

        // Project slots are always directly after the gamma value, so we can write to them by
        // adding 1 to the gamma id and the index
        for result_idx in 0..num_results {
            let phi_type = region_results[0].0[result_idx].get_type();
            let phi = llvm_builder.builder.build_phi(phi_type, "gamma.phi")?;
            let incoming: Vec<_> = region_results
                .iter()
                .map(|(vals, bb)| (&vals[result_idx] as &dyn BasicValue, *bb))
                .collect();
            phi.add_incoming(&incoming);

            let project_id = ValueId(gamma_value_id.0 + 1 + result_idx as u32);
            mapper.set_val(project_id, phi.as_basic_value());
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, Linkage, RVSDGMod,
        builder::BranchResult,
        func::FnResult,
        lower_to_llvm::test_utils::test_utils::jit_run_i32,
        types::{BOOL, I32},
        value::ConstValue,
    };

    #[test]
    fn gamma_true_branch() {
        // if true { 42 } else { 99 } => 42
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(1));
            let res = rb.gamma(
                cond,
                state,
                &[],
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(42)],
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

        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn gamma_false_branch() {
        // if false { 42 } else { 99 } => 99
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(0));
            let res = rb.gamma(
                cond,
                state,
                &[],
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(42)],
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

        assert_eq!(jit_run_i32(&rvsdg, "test"), 99);
    }

    #[test]
    fn gamma_nested_arithmetic() {
        // if true { 10 + 20 } else { 0 } => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(1));
            let res = rb.gamma(
                cond,
                state,
                &[],
                |rb| {
                    let a = rb.const_i32(10);
                    let b = rb.const_i32(20);
                    let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
                    BranchResult {
                        state,
                        values: vec![sum],
                    }
                },
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(0)],
                },
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gamma_computed_condition() {
        // x=5, y=3; if x > y { 1 } else { 0 } => 1
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(5);
            let y = rb.const_i32(3);
            let cond = rb.icmp(ICmpPred::SignedGt, x, y);
            let res = rb.gamma(
                cond,
                state,
                &[],
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(1)],
                },
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(0)],
                },
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn gamma_n_switch_case_0() {
        // switch(0) { case 0: 10, case 1: 20, case 2: 30 } => 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.const_i32(0);
            let res = rb.gamma_n(
                cond,
                state,
                &[],
                &[
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(10)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(20)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(30)],
                    },
                ],
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn gamma_n_switch_case_1() {
        // switch(1) { case 0: 10, case 1: 20, case 2: 30 } => 20
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.const_i32(1);
            let res = rb.gamma_n(
                cond,
                state,
                &[],
                &[
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(10)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(20)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(30)],
                    },
                ],
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 20);
    }

    #[test]
    fn gamma_n_switch_case_2() {
        // switch(2) { case 0: 10, case 1: 20, case 2: 30 } => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.const_i32(2);
            let res = rb.gamma_n(
                cond,
                state,
                &[],
                &[
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(10)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(20)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(30)],
                    },
                ],
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gamma_n_switch_with_arithmetic() {
        // switch(1) { case 0: 100, case 1: 3*7, case 2: 0, case 3: -1 } => 21
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.const_i32(1);
            let res = rb.gamma_n(
                cond,
                state,
                &[],
                &[
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(100)],
                    },
                    &|rb| {
                        let a = rb.const_i32(3);
                        let b = rb.const_i32(7);
                        let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                        BranchResult {
                            state,
                            values: vec![product],
                        }
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(0)],
                    },
                    &|rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(-1)],
                    },
                ],
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 21);
    }

    #[test]
    fn gamma_with_inputs() {
        // a=10, b=20; if true { a + b } else { a - b } => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(10);
            let b = rb.const_i32(20);
            let cond = rb.constant(BOOL, ConstValue::Int(1));
            let res = rb.gamma(
                cond,
                state,
                &[a, b],
                |rb| {
                    let x = rb.param(0);
                    let y = rb.param(1);
                    let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
                    BranchResult {
                        state,
                        values: vec![sum],
                    }
                },
                |rb| {
                    let x = rb.param(0);
                    let y = rb.param(1);
                    let diff = rb.binary(BinaryOp::Sub, ArithFlags::default(), x, y, I32);
                    BranchResult {
                        state,
                        values: vec![diff],
                    }
                },
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gamma_n_with_inputs() {
        // switch(2) over 3 branches, each using inputs a=10 b=20:
        // case 0: a + b (30), case 1: a - b (-10), case 2: a * b (200) => 200
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(10);
            let b = rb.const_i32(20);
            let cond = rb.const_i32(2);
            let res = rb.gamma_n(
                cond,
                state,
                &[a, b],
                &[
                    &|rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
                        BranchResult {
                            state,
                            values: vec![sum],
                        }
                    },
                    &|rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let diff = rb.binary(BinaryOp::Sub, ArithFlags::default(), x, y, I32);
                        BranchResult {
                            state,
                            values: vec![diff],
                        }
                    },
                    &|rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, y, I32);
                        BranchResult {
                            state,
                            values: vec![product],
                        }
                    },
                ],
            );
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 200);
    }

    #[test]
    fn gamma_multiple_results() {
        // if true { (10, 20) } else { (1, 2) }
        // return result_0 * result_1 = 10 * 20 = 200
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(1));
            let res = rb.gamma(
                cond,
                state,
                &[],
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(10), rb.const_i32(20)],
                },
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(1), rb.const_i32(2)],
                },
            );
            let a = res.result(0);
            let b = res.result(1);
            let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
            FnResult {
                state: res.state,
                values: vec![product],
            }
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 200);
    }
}
