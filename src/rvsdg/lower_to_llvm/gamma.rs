use crate::rvsdg::{
    RVSDGMod, RegionsSpan, ValueId, ValuesSpan,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
};
use color_eyre::eyre::{bail, eyre};
use inkwell::{
    basic_block::BasicBlock,
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
    ) -> color_eyre::Result<Option<BasicValueEnum<'ctx>>> {
        let cond = self.expect_value(llvm_builder, mapper, rvsdg_func, condition)?;
        let region_ids = self.region_pool.get(regions).to_vec();
        let input_ids = self.value_pool.get(inputs).to_vec();
        let func = mapper.get_fn(rvsdg_func.id).ok_or_else(|| {
            eyre!(
                "function `{}` was not registered before lowering its gamma",
                rvsdg_func.name
            )
        })?;

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

        // Emit the branch from the current block. The choice of branch vs
        // switch is driven by the condition TYPE, not the arm count: an
        // `i1` condition is a two-way predicate (true -> region 0, false ->
        // region 1), while any wider integer is an arm INDEX (value k
        // selects region k, with region 0 as the switch default). Arm count
        // alone can't disambiguate -- a 2-arm gamma can come from a switch
        // with one case plus default, whose selector is an i32 index, not a
        // bool.
        let cond_int = cond.into_int_value();
        if cond_int.get_type().get_bit_width() == 1 {
            debug_assert_eq!(
                region_bbs.len(),
                2,
                "an i1-conditioned gamma must have exactly two regions"
            );
            llvm_builder.builder.build_conditional_branch(
                cond_int,
                region_bbs[0],
                region_bbs[1],
            )?;
        } else {
            // The first block is the switch default; cases 1..N map value
            // `i` to region `i`.
            let cases: Vec<_> = region_bbs
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &bb)| (cond_int.get_type().const_int(i as u64, false), bb))
                .collect();
            llvm_builder
                .builder
                .build_switch(cond_int, region_bbs[0], &cases)?;
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
            let params = region.params.clone();
            for (j, &param_id) in params.iter().enumerate() {
                if j < input_ids.len() {
                    let input_val =
                        self.expect_value(llvm_builder, mapper, rvsdg_func, input_ids[j])?;
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
            let actual_bb = llvm_builder
                .builder
                .get_insert_block()
                .ok_or_else(|| eyre!("gamma arm ended with no current basic block"))?;
            llvm_builder.builder.build_unconditional_branch(merge_bb)?;
            region_results.push((results, actual_bb));
        }

        // Build phi nodes in the merge block and write results to Project slots
        llvm_builder.builder.position_at_end(merge_bb);
        let num_results = region_results.first().map(|(r, _)| r.len()).unwrap_or(0);

        // Every gamma region must produce the same number of value-results; the
        // phi-per-result wiring below indexes each region by `result_idx` and
        // would otherwise read out of bounds (or silently mis-wire) if a region
        // lowered a different count.
        if let Some((mismatched, _)) = region_results.iter().find(|(r, _)| r.len() != num_results) {
            bail!(
                "gamma regions produced mismatched result arities: expected {num_results}, \
                 found a region with {}",
                mismatched.len()
            );
        }

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
        ArithFlags, BinaryOp, ICmpPred, Linkage, MatchArm, RVSDGMod,
        builder::BranchResult,
        func::FnResult,
        lower_to_llvm::test_utils::test_utils::jit_run_i32,
        types::{BOOL, I32},
        value::ConstValue,
    };

    #[test]
    fn gamma_via_match_predicate_selects_arm() {
        // A control predicate produced by `match` drives the gamma. Input 1 is
        // matched to alternative 1, selecting the second arm: 20.
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let input = rb.const_i32(1);
            let pred = rb.match_op(
                input,
                &[
                    MatchArm {
                        value: 0,
                        alternative: 0,
                    },
                    MatchArm {
                        value: 1,
                        alternative: 1,
                    },
                    MatchArm {
                        value: 2,
                        alternative: 2,
                    },
                ],
                0,
                3,
            );
            let res = rb.gamma_n(
                pred,
                state,
                &[],
                &[
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(10)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(20)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(30)],
                        })
                    },
                ],
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 20);
    }

    #[test]
    fn gamma_via_match_two_way_from_bool() {
        // A 2-alternative control predicate from an i1: `match` maps true(1) to
        // alternative 0 and everything else to the default 1. The condition
        // `5 > 3` is true, so alternative 0 (the first arm) is selected: 42.
        // This exercises the control predicate's switch-path lowering (value k
        // -> region k), the opposite convention from the raw-i1 gamma path.
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(5);
            let y = rb.const_i32(3);
            let cond = rb.icmp(ICmpPred::SignedGt, x, y);
            let pred = rb.match_op(
                cond,
                &[MatchArm {
                    value: 1,
                    alternative: 0,
                }],
                1,
                2,
            );
            let res = rb.gamma(
                pred,
                state,
                &[],
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(42)],
                    })
                },
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(99)],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

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
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(42)],
                    })
                },
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(99)],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(42)],
                    })
                },
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(99)],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                    Ok(BranchResult {
                        state,
                        values: vec![sum],
                    })
                },
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(0)],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(1)],
                    })
                },
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(0)],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(10)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(20)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(30)],
                        })
                    },
                ],
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(10)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(20)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(30)],
                        })
                    },
                ],
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(10)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(20)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(30)],
                        })
                    },
                ],
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(100)],
                        })
                    },
                    &|rb| {
                        let a = rb.const_i32(3);
                        let b = rb.const_i32(7);
                        let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                        Ok(BranchResult {
                            state,
                            values: vec![product],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(0)],
                        })
                    },
                    &|rb| {
                        Ok(BranchResult {
                            state,
                            values: vec![rb.const_i32(-1)],
                        })
                    },
                ],
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                    Ok(BranchResult {
                        state,
                        values: vec![sum],
                    })
                },
                |rb| {
                    let x = rb.param(0);
                    let y = rb.param(1);
                    let diff = rb.binary(BinaryOp::Sub, ArithFlags::default(), x, y, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![diff],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                        Ok(BranchResult {
                            state,
                            values: vec![sum],
                        })
                    },
                    &|rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let diff = rb.binary(BinaryOp::Sub, ArithFlags::default(), x, y, I32);
                        Ok(BranchResult {
                            state,
                            values: vec![diff],
                        })
                    },
                    &|rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, y, I32);
                        Ok(BranchResult {
                            state,
                            values: vec![product],
                        })
                    },
                ],
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
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
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(10), rb.const_i32(20)],
                    })
                },
                |rb| {
                    Ok(BranchResult {
                        state,
                        values: vec![rb.const_i32(1), rb.const_i32(2)],
                    })
                },
            )?;
            let a = res.result(0);
            let b = res.result(1);
            let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
            Ok(FnResult {
                state: res.state,
                values: vec![product],
            })
        });

        assert_eq!(jit_run_i32(&rvsdg, "test"), 200);
    }
}
