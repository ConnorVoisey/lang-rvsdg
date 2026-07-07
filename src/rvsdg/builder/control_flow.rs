use color_eyre::eyre::eyre;

use crate::rvsdg::{
    FuncId, MatchArm, RegionId, State, Value, ValueId, ValueKind,
    func::CallResult,
    types::{FuncTypeId, ScalarType, TypeRef, VOID},
};

use super::{
    BranchResult, GammaResult, LoopResult, PhiBody, PhiResult, RegionBuilder, ThetaResult,
};

impl<'a> RegionBuilder<'a> {
    /// N-way conditional branch. Condition value selects which region executes:
    /// 0 -> first branch, 1 -> second, etc. All branches must return the same
    /// number and types of values.
    // (20-50+ in real programs). Profile real-world code to determine if an
    // incremental builder API (gamma_begin/add_input/build) would be worthwhile.
    #[inline]
    pub fn gamma_n(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: &[ValueId],
        branches: &[&dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>],
    ) -> color_eyre::Result<GammaResult> {
        debug_assert!(branches.len() >= 2, "gamma requires at least 2 branches");
        let inputs_span = self.graph.value_pool.push_slice(inputs);

        let mut branch_regions: Vec<RegionId> = Vec::with_capacity(branches.len());
        let mut result_count: Option<u16> = None;

        let param_types: Vec<TypeRef> = inputs
            .iter()
            .map(|&id| self.graph.values[id.0 as usize].ty)
            .collect();

        for branch in branches {
            let mut rb = RegionBuilder::new_with_params(self.graph, state, &param_types);
            let res = branch(&mut rb)?;
            let count = res.values.len() as u16;
            match result_count {
                None => result_count = Some(count),
                Some(expected) => debug_assert_eq!(
                    count, expected,
                    "all gamma branches must return the same number of values"
                ),
            }
            let results = rb.graph.value_pool.push_slice(&res.values);
            rb.graph.regions[rb.region_id.0 as usize].results = results;
            branch_regions.push(rb.region_id);
        }

        let result_count =
            result_count.ok_or_else(|| eyre!("gamma_n requires at least one branch"))?;
        let regions = self.graph.region_pool.push_slice(&branch_regions);

        let gamma_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Gamma {
                condition,
                inputs: inputs_span,
                state,
                regions,
            },
        });
        let out_state = State(gamma_val);

        let first_result = ValueId(self.graph.values.len() as u32);
        let first_region = branch_regions[0];
        let first_results = self.graph.regions[first_region.0 as usize].results;
        for i in 0..result_count {
            let ty = self.graph.values
                [self.graph.value_pool.get(first_results)[i as usize].0 as usize]
                .ty;
            self.add_value(Value {
                ty,
                kind: ValueKind::Project {
                    call: gamma_val,
                    index: i,
                },
            });
        }

        Ok(GammaResult {
            state: out_state,
            first_result,
            result_count,
        })
    }

    /// Assemble a gamma node from regions that were built incrementally
    /// (the paper's process-first, assemble-afterwards order: subregions
    /// are emitted with capture-on-demand parameters, THEN the node is
    /// created). Every region's parameter list must already be aligned to
    /// `inputs` positionally, and its results set, all with `result_count`
    /// entries.
    pub fn finish_gamma(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: &[ValueId],
        branch_regions: &[RegionId],
        result_count: u16,
    ) -> GammaResult {
        debug_assert!(
            branch_regions.len() >= 2,
            "gamma requires at least 2 branches"
        );
        let inputs_span = self.graph.value_pool.push_slice(inputs);
        let regions = self.graph.region_pool.push_slice(branch_regions);

        let gamma_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Gamma {
                condition,
                inputs: inputs_span,
                state,
                regions,
            },
        });
        let out_state = State(gamma_val);

        let first_result = ValueId(self.graph.values.len() as u32);
        let first_region = branch_regions[0];
        let first_results = self.graph.regions[first_region.0 as usize].results;
        for i in 0..result_count {
            let ty = self.graph.values
                [self.graph.value_pool.get(first_results)[i as usize].0 as usize]
                .ty;
            self.add_value(Value {
                ty,
                kind: ValueKind::Project {
                    call: gamma_val,
                    index: i,
                },
            });
        }

        GammaResult {
            state: out_state,
            first_result,
            result_count,
        }
    }

    /// Assemble a theta node from a body region that was built
    /// incrementally (see `finish_gamma`). The region's parameters must be
    /// aligned to `loop_vars` positionally and its results (the
    /// next-iteration values) set; `condition` is the repetition predicate,
    /// a value inside the body region (alternative 1 repeats).
    pub fn finish_theta(
        &mut self,
        state: State,
        loop_vars: &[ValueId],
        body_region: RegionId,
        condition: ValueId,
    ) -> ThetaResult {
        let loop_span = self.graph.value_pool.push_slice(loop_vars);
        let result_count = loop_vars.len() as u16;

        let theta_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Theta {
                loop_vars: loop_span,
                condition,
                state,
                region_id: body_region,
            },
        });
        let out_state = State(theta_val);

        let first_result = ValueId(self.graph.values.len() as u32);
        for i in 0..result_count {
            let ty = self.graph.values[loop_vars[i as usize].0 as usize].ty;
            self.add_value(Value {
                ty,
                kind: ValueKind::Project {
                    call: theta_val,
                    index: i,
                },
            });
        }

        ThetaResult {
            state: out_state,
            first_result,
            result_count,
        }
    }

    /// Two-way if/else convenience. Condition is a bool: true -> first branch,
    /// false -> second branch. See `gamma_n` for the inputs allocation note.
    #[inline]
    pub fn gamma(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: &[ValueId],
        true_branch: impl Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>,
        false_branch: impl Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>,
    ) -> color_eyre::Result<GammaResult> {
        self.gamma_n(condition, state, inputs, &[&true_branch, &false_branch])
    }

    /// Build a `match` node: convert integer `input` into a control/predicate
    /// value with `alternatives` alternatives (see [`ValueKind::Match`]). Each
    /// `(value, alternative)` arm maps an input value to a control alternative;
    /// any input value not listed maps to `default`. The result is a
    /// control-typed value (`TypeRef::Control(alternatives)`) suitable as a gamma
    /// decision predicate or a theta repetition predicate -- the typed predicate the
    /// paper's gamma/theta consume rather than a raw integer (section 2.2).
    #[inline]
    pub fn match_op(
        &mut self,
        input: ValueId,
        arms: &[MatchArm],
        default: u32,
        alternatives: u32,
    ) -> ValueId {
        let arms_span = self.graph.match_arm_pool.push_slice(arms);
        self.add_value(Value {
            ty: TypeRef::Control(alternatives),
            kind: ValueKind::Match {
                input,
                arms: arms_span,
                default,
                alternatives,
            },
        })
    }

    /// Control predicate for an LLVM `i1` condition: `true` (1) selects arm 0,
    /// anything else (the default) selects arm 1 -- matching `CondBr`'s
    /// `[true_dest, false_dest]` arm order.
    pub fn bool_predicate(&mut self, condition: ValueId) -> ValueId {
        self.match_op(
            condition,
            &[MatchArm {
                value: 1,
                alternative: 0,
            }],
            1,
            2,
        )
    }

    /// Control predicate over an already-0-based index `q` for an `n`-way demux:
    /// value `k` selects arm `k` (the identity mapping), default arm 0.
    pub fn identity_match(&mut self, q: ValueId, n: u32) -> ValueId {
        let arms: Vec<MatchArm> = (0..n)
            .map(|k| MatchArm {
                value: k as i64,
                alternative: k,
            })
            .collect();
        self.match_op(q, &arms, 0, n)
    }

    #[inline]
    pub fn theta(
        &mut self,
        state: State,
        loop_vars: &[ValueId],
        loop_body: impl FnOnce(&mut RegionBuilder) -> color_eyre::Result<LoopResult>,
    ) -> color_eyre::Result<ThetaResult> {
        let loop_span = self.graph.value_pool.push_slice(loop_vars);
        let result_count = loop_vars.len() as u16;

        let param_types: Vec<TypeRef> = loop_vars
            .iter()
            .map(|&id| self.graph.values[id.0 as usize].ty)
            .collect();

        let (region, condition) = {
            let mut rb = RegionBuilder::new_with_params(self.graph, state, &param_types);
            let res = loop_body(&mut rb)?;
            debug_assert_eq!(
                res.next_vars.len() as u16,
                result_count,
                "theta body must return the same number of loop vars"
            );
            let results = rb.graph.value_pool.push_slice(&res.next_vars);
            rb.graph.regions[rb.region_id.0 as usize].results = results;
            (rb.region_id, res.condition)
        };

        let theta_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Theta {
                loop_vars: loop_span,
                condition,
                state,
                region_id: region,
            },
        });
        let out_state = State(theta_val);

        let first_result = ValueId(self.graph.values.len() as u32);
        for i in 0..result_count {
            let ty = self.graph.values[loop_vars[i as usize].0 as usize].ty;
            self.add_value(Value {
                ty,
                kind: ValueKind::Project {
                    call: theta_val,
                    index: i,
                },
            });
        }

        Ok(ThetaResult {
            state: out_state,
            first_result,
            result_count,
        })
    }

    /// Build a phi node for mutually recursive function definitions.
    ///
    /// `rv_count` is the number of recursion variables (one per mutually recursive function).
    /// The closure receives a `RegionBuilder` whose first `rv_count` params are the recursion
    /// variables -- handles the body can use to refer to the functions being defined.
    /// The closure must return `PhiBody` containing the lambda values produced inside.
    #[inline]
    pub fn phi(
        &mut self,
        state: State,
        rv_count: u16,
        body: impl FnOnce(&mut RegionBuilder, &[ValueId]) -> PhiBody,
    ) -> PhiResult {
        let mut rb = RegionBuilder::new_empty(self.graph, state);
        let region = rb.region_id;

        // Create recursion variable params inside the phi region
        let rv_start = ValueId(rb.graph.values.len() as u32);
        for i in 0..rv_count {
            let id = ValueId(rb.graph.values.len() as u32);
            rb.graph.values.push(Value {
                ty: TypeRef::Scalar(ScalarType::Void),
                kind: ValueKind::RegionParam {
                    index: i as u32,
                    ty: TypeRef::Scalar(ScalarType::Void),
                },
            });
            rb.graph.regions[region.0 as usize].nodes.push(id);
        }

        let rvs: Vec<ValueId> = (0..rv_count)
            .map(|i| ValueId(rv_start.0 + i as u32))
            .collect();

        let phi_body = body(&mut rb, &rvs);
        debug_assert_eq!(
            phi_body.values.len() as u16,
            rv_count,
            "phi body must return exactly one lambda per recursion variable"
        );
        let results = rb.graph.value_pool.push_slice(&phi_body.values);
        rb.graph.regions[region.0 as usize].results = results;

        let phi_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Phi { region, rv_count },
        });

        let first_result = ValueId(self.graph.values.len() as u32);
        for i in 0..rv_count {
            self.add_value(Value {
                ty: TypeRef::Scalar(ScalarType::Void),
                kind: ValueKind::Project {
                    call: phi_val,
                    index: i,
                },
            });
        }

        PhiResult {
            first_result,
            result_count: rv_count,
        }
    }

    #[inline]
    pub fn call(&mut self, fn_id: FuncId, state: State, args: &[ValueId]) -> CallResult {
        let args_span = self.graph.value_pool.push_slice(args);

        // call value is the state node
        let call_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Call {
                state,
                fn_id,
                args: args_span,
            },
        });
        let out_state = State(call_val);

        let first_res = ValueId(self.graph.values.len() as u32);
        let result_count = self.graph.functions[fn_id.0 as usize].return_types.len() as u16;
        for i in 0..result_count {
            let ty = self.graph.functions[fn_id.0 as usize].return_types[i as usize];
            self.add_value(Value {
                ty,
                kind: ValueKind::Project {
                    call: call_val,
                    index: i,
                },
            });
        }

        CallResult {
            state: out_state,
            first_result: first_res,
            result_count,
        }
    }

    /// Call through a function pointer. The caller must provide the callee's
    /// full signature (interned in the type arena): it can't be looked up
    /// from the function table, and the callee value's type is an opaque
    /// pointer. The result projections come from the signature's return
    /// type.
    #[inline]
    pub fn call_indirect(
        &mut self,
        callee: ValueId,
        state: State,
        args: &[ValueId],
        fn_ty: FuncTypeId,
    ) -> CallResult {
        let args_span = self.graph.value_pool.push_slice(args);

        let call_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::CallIndirect {
                state,
                callee,
                fn_ty,
                args: args_span,
            },
        });
        let out_state = State(call_val);

        let first_res = ValueId(self.graph.values.len() as u32);
        let ret = self.graph.types.get_fn(fn_ty).ret;
        let result_count = u16::from(ret != VOID);
        for i in 0..result_count {
            self.add_value(Value {
                ty: ret,
                kind: ValueKind::Project {
                    call: call_val,
                    index: i,
                },
            });
        }

        CallResult {
            state: out_state,
            first_result: first_res,
            result_count,
        }
    }
}
