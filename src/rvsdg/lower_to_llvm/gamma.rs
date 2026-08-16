use crate::rvsdg::{RegionsSpan, ValueId, ValuesSpan, lower_to_llvm::FunctionLowerer};
use color_eyre::eyre::{bail, eyre};
use inkwell::{
    basic_block::BasicBlock,
    values::{BasicValue, BasicValueEnum},
};

impl<'m, 'a, 'ctx> FunctionLowerer<'m, 'a, 'ctx> {
    #[inline]
    pub(crate) fn lower_gamma(
        &mut self,
        gamma_value_id: ValueId,
        condition: ValueId,
        inputs: ValuesSpan,
        regions: RegionsSpan,
    ) -> color_eyre::Result<Option<BasicValueEnum<'ctx>>> {
        let cond = self.expect_value(condition)?;
        let graph = self.graph;
        let region_ids = graph.region_pool.get(regions);
        let input_ids = graph.value_pool.get(inputs);
        let func = self.mod_lower.get_fn(self.func_id).ok_or_else(|| {
            eyre!(
                "function `{}` was not registered before lowering its gamma",
                self.function().name
            )
        })?;

        // Create all basic blocks upfront
        let merge_bb = self
            .mod_lower
            .context
            .append_basic_block(func, "gamma.merge");
        let region_bbs: Vec<_> = region_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                self.mod_lower
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
            self.builder
                .build_conditional_branch(cond_int, region_bbs[0], region_bbs[1])?;
        } else {
            // The first block is the switch default; cases 1..N map value
            // `i` to region `i`.
            let cases: Vec<_> = region_bbs
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &bb)| (cond_int.get_type().const_int(i as u64, false), bb))
                .collect();
            self.builder.build_switch(cond_int, region_bbs[0], &cases)?;
        }

        // Lower each region, collecting (result_values, basic_block) per region
        // This is a vec of vecs and will be slow, consider replacing this with something more
        // efficient in the future
        let mut region_results: Vec<(Vec<BasicValueEnum<'ctx>>, BasicBlock<'ctx>)> =
            Vec::with_capacity(region_ids.len());
        for (i, &region_id) in region_ids.iter().enumerate() {
            let bb = region_bbs[i];
            self.builder.position_at_end(bb);

            // Bind gamma inputs to this region's params
            for (j, &param_id) in graph.region_params(region_id).iter().enumerate() {
                if j < input_ids.len() {
                    let input_val = self.expect_value(input_ids[j])?;
                    self.set_val(param_id, input_val);
                }
            }

            self.lower_region(region_id)?;

            // Collect result values from the region
            let result_ids = graph.region_results(region_id);
            let mut results: Vec<BasicValueEnum<'ctx>> = Vec::with_capacity(result_ids.len());
            for &rid in result_ids {
                if let Some(value) = self.lowered_result(rid)? {
                    results.push(value);
                }
            }

            // lowering the region could insert a new basic block, so get the current basic block
            // here
            let actual_bb = self
                .builder
                .get_insert_block()
                .ok_or_else(|| eyre!("gamma arm ended with no current basic block"))?;
            self.builder.build_unconditional_branch(merge_bb)?;
            region_results.push((results, actual_bb));
        }

        // Build phi nodes in the merge block and write results to Project slots
        self.builder.position_at_end(merge_bb);
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
            let phi = self.builder.build_phi(phi_type, "gamma.phi")?;
            let incoming: Vec<_> = region_results
                .iter()
                .map(|(vals, bb)| (&vals[result_idx] as &dyn BasicValue, *bb))
                .collect();
            phi.add_incoming(&incoming);

            let project_id = graph.projection_of(gamma_value_id, result_idx as u16);
            self.set_val(project_id, phi.as_basic_value());
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, Linkage, MatchArm, RVSDGMod,
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
        rvsdg
            .define_fn(func_id, |rb| {
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
                    &[],
                    &[
                        &|rb| Ok(vec![rb.const_i32(10)]),
                        &|rb| Ok(vec![rb.const_i32(20)]),
                        &|rb| Ok(vec![rb.const_i32(30)]),
                    ],
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

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
        rvsdg
            .define_fn(func_id, |rb| {
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
                    &[],
                    |rb| Ok(vec![rb.const_i32(42)]),
                    |rb| Ok(vec![rb.const_i32(99)]),
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn gamma_true_branch() {
        // if true { 42 } else { 99 } => 42
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.constant(BOOL, ConstValue::Int(1));
                let res = rb.gamma(
                    cond,
                    &[],
                    |rb| Ok(vec![rb.const_i32(42)]),
                    |rb| Ok(vec![rb.const_i32(99)]),
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn gamma_false_branch() {
        // if false { 42 } else { 99 } => 99
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.constant(BOOL, ConstValue::Int(0));
                let res = rb.gamma(
                    cond,
                    &[],
                    |rb| Ok(vec![rb.const_i32(42)]),
                    |rb| Ok(vec![rb.const_i32(99)]),
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 99);
    }

    #[test]
    fn gamma_nested_arithmetic() {
        // if true { 10 + 20 } else { 0 } => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.constant(BOOL, ConstValue::Int(1));
                let res = rb.gamma(
                    cond,
                    &[],
                    |rb| {
                        let a = rb.const_i32(10);
                        let b = rb.const_i32(20);
                        let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
                        Ok(vec![sum])
                    },
                    |rb| Ok(vec![rb.const_i32(0)]),
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gamma_computed_condition() {
        // x=5, y=3; if x > y { 1 } else { 0 } => 1
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let x = rb.const_i32(5);
                let y = rb.const_i32(3);
                let cond = rb.icmp(ICmpPred::SignedGt, x, y);
                let res = rb.gamma(
                    cond,
                    &[],
                    |rb| Ok(vec![rb.const_i32(1)]),
                    |rb| Ok(vec![rb.const_i32(0)]),
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn gamma_n_switch_case_0() {
        // switch(0) { case 0: 10, case 1: 20, case 2: 30 } => 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.const_i32(0);
                let res = rb.gamma_n(
                    cond,
                    &[],
                    &[
                        &|rb| Ok(vec![rb.const_i32(10)]),
                        &|rb| Ok(vec![rb.const_i32(20)]),
                        &|rb| Ok(vec![rb.const_i32(30)]),
                    ],
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn gamma_n_switch_case_1() {
        // switch(1) { case 0: 10, case 1: 20, case 2: 30 } => 20
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.const_i32(1);
                let res = rb.gamma_n(
                    cond,
                    &[],
                    &[
                        &|rb| Ok(vec![rb.const_i32(10)]),
                        &|rb| Ok(vec![rb.const_i32(20)]),
                        &|rb| Ok(vec![rb.const_i32(30)]),
                    ],
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 20);
    }

    #[test]
    fn gamma_n_switch_case_2() {
        // switch(2) { case 0: 10, case 1: 20, case 2: 30 } => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.const_i32(2);
                let res = rb.gamma_n(
                    cond,
                    &[],
                    &[
                        &|rb| Ok(vec![rb.const_i32(10)]),
                        &|rb| Ok(vec![rb.const_i32(20)]),
                        &|rb| Ok(vec![rb.const_i32(30)]),
                    ],
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gamma_n_switch_with_arithmetic() {
        // switch(1) { case 0: 100, case 1: 3*7, case 2: 0, case 3: -1 } => 21
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.const_i32(1);
                let res = rb.gamma_n(
                    cond,
                    &[],
                    &[
                        &|rb| Ok(vec![rb.const_i32(100)]),
                        &|rb| {
                            let a = rb.const_i32(3);
                            let b = rb.const_i32(7);
                            let product =
                                rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                            Ok(vec![product])
                        },
                        &|rb| Ok(vec![rb.const_i32(0)]),
                        &|rb| Ok(vec![rb.const_i32(-1)]),
                    ],
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 21);
    }

    #[test]
    fn gamma_with_inputs() {
        // a=10, b=20; if true { a + b } else { a - b } => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(10);
                let b = rb.const_i32(20);
                let cond = rb.constant(BOOL, ConstValue::Int(1));
                let res = rb.gamma(
                    cond,
                    &[a, b],
                    |rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
                        Ok(vec![sum])
                    },
                    |rb| {
                        let x = rb.param(0);
                        let y = rb.param(1);
                        let diff = rb.binary(BinaryOp::Sub, ArithFlags::default(), x, y, I32);
                        Ok(vec![diff])
                    },
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gamma_n_with_inputs() {
        // switch(2) over 3 branches, each using inputs a=10 b=20:
        // case 0: a + b (30), case 1: a - b (-10), case 2: a * b (200) => 200
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(10);
                let b = rb.const_i32(20);
                let cond = rb.const_i32(2);
                let res = rb.gamma_n(
                    cond,
                    &[a, b],
                    &[
                        &|rb| {
                            let x = rb.param(0);
                            let y = rb.param(1);
                            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
                            Ok(vec![sum])
                        },
                        &|rb| {
                            let x = rb.param(0);
                            let y = rb.param(1);
                            let diff = rb.binary(BinaryOp::Sub, ArithFlags::default(), x, y, I32);
                            Ok(vec![diff])
                        },
                        &|rb| {
                            let x = rb.param(0);
                            let y = rb.param(1);
                            let product =
                                rb.binary(BinaryOp::Mul, ArithFlags::default(), x, y, I32);
                            Ok(vec![product])
                        },
                    ],
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 200);
    }

    #[test]
    fn gamma_multiple_results() {
        // if true { (10, 20) } else { (1, 2) }
        // return result_0 * result_1 = 10 * 20 = 200
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let cond = rb.constant(BOOL, ConstValue::Int(1));
                let res = rb.gamma(
                    cond,
                    &[],
                    |rb| Ok(vec![rb.const_i32(10), rb.const_i32(20)]),
                    |rb| Ok(vec![rb.const_i32(1), rb.const_i32(2)]),
                )?;
                let a = res.result(0);
                let b = res.result(1);
                let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                Ok(vec![product])
            })
            .unwrap();

        assert_eq!(jit_run_i32(&rvsdg, "test"), 200);
    }
}
