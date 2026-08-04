use color_eyre::eyre::eyre;

use crate::rvsdg::{
    FuncId, MatchArm, RegionId, State, Value, ValueId, ValueKind,
    func::{CallResult, SignatureId},
    types::{ScalarType, TypeRef, VOID},
};

use super::{BranchResult, GammaResult, LoopResult, RegionBuilder, ThetaResult};

impl<'a> RegionBuilder<'a> {
    /// N-way conditional branch. Condition value selects which region executes:
    /// 0 -> first branch, 1 -> second, etc. All branches must return the same
    /// number and types of values.
    #[inline]
    pub fn gamma_n(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: &[ValueId],
        branches: &[&dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>],
    ) -> color_eyre::Result<GammaResult> {
        debug_assert!(branches.len() >= 2, "gamma requires at least 2 branches");
        let mut branch_regions: Vec<RegionId> = Vec::with_capacity(branches.len());
        let mut result_count: Option<u16> = None;

        let param_types: Vec<TypeRef> = inputs
            .iter()
            .map(|&id| *self.graph.get_value_type(id))
            .collect();

        for branch in branches {
            let mut rb =
                RegionBuilder::new_with_params(self.graph, self.module_tables, state, &param_types);
            let res = branch(&mut rb)?;
            let count = res.values.len() as u16;
            match result_count {
                None => result_count = Some(count),
                Some(expected) => debug_assert_eq!(
                    count, expected,
                    "all gamma branches must return the same number of values"
                ),
            }
            rb.graph.seal_region(rb.region_id, &res.values, res.state);
            branch_regions.push(rb.region_id);
        }

        let result_count =
            result_count.ok_or_else(|| eyre!("gamma requires at least 2 branches"))?;
        Ok(self.finish_gamma(condition, state, inputs, &branch_regions, result_count))
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
        for &arm in branch_regions {
            self.graph.get_region_mut(arm).owner = gamma_val;
        }
        let out_state = State(gamma_val);

        let first_result = ValueId(self.graph.value_kinds.len() as u32);
        let first_region = branch_regions[0];
        for i in 0..result_count {
            let ty = *self
                .graph
                .get_value_type(self.graph.region_results(first_region)[i as usize]);
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
        self.graph.get_region_mut(body_region).owner = theta_val;
        let out_state = State(theta_val);

        let first_result = ValueId(self.graph.value_kinds.len() as u32);
        for i in 0..result_count {
            let ty = *self.graph.get_value_type(loop_vars[i as usize]);
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
    /// false -> second branch.
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
        let param_types: Vec<TypeRef> = loop_vars
            .iter()
            .map(|id| *self.graph.get_value_type(*id))
            .collect();

        let (region, condition) = {
            let mut rb =
                RegionBuilder::new_with_params(self.graph, self.module_tables, state, &param_types);
            let res = loop_body(&mut rb)?;
            debug_assert_eq!(
                res.next_vars.len(),
                loop_vars.len(),
                "theta body must return the same number of loop vars"
            );
            rb.graph
                .seal_region(rb.region_id, &res.next_vars, res.next_state);
            (rb.region_id, res.condition)
        };

        Ok(self.finish_theta(state, loop_vars, region, condition))
    }

    /// Call a known function with an ABI signature derived from its
    /// declaration. Exact for hand-built graphs and any non-variadic
    /// call; a parser lowering real call sites should use
    /// [`call_with_signature`](Self::call_with_signature) with the site's
    /// own signature instead, because a variadic call site carries ABI
    /// attributes for actual arguments the declaration knows nothing
    /// about.
    #[inline]
    pub fn call(&mut self, fn_id: FuncId, state: State, args: &[ValueId]) -> CallResult {
        let sig =
            self.module_tables.get_function(fn_id).declared_sig.expect(
                "multi-return function has no declaration signature; use call_with_signature",
            );
        self.call_with_signature(fn_id, state, args, sig)
    }

    /// Call a known function with the call site's own interned ABI
    /// signature (one attribute set per ACTUAL argument, return
    /// attributes, calling convention). LLVM attributes live on call
    /// sites as well as declarations, so the site's signature is the
    /// source of truth at emission.
    #[inline]
    pub fn call_with_signature(
        &mut self,
        fn_id: FuncId,
        state: State,
        args: &[ValueId],
        sig: SignatureId,
    ) -> CallResult {
        let args_span = self.graph.value_pool.push_slice(args);

        // call value is the state node
        let call_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Call {
                state,
                fn_id,
                sig,
                args: args_span,
            },
        });
        let out_state = State(call_val);

        let first_res = ValueId(self.graph.value_kinds.len() as u32);
        let result_count = self.module_tables.get_function(fn_id).return_types.len() as u16;
        for i in 0..result_count {
            let ty = self.module_tables.get_function(fn_id).return_types[i as usize];
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

    /// Call through a function pointer. The caller must provide the call
    /// site's interned [`Signature`](crate::rvsdg::func::Signature): it
    /// can't be looked up from the function table, and the callee value's
    /// type is an opaque pointer. The result projections come from the
    /// signature's return type.
    #[inline]
    pub fn call_indirect(
        &mut self,
        callee: ValueId,
        state: State,
        args: &[ValueId],
        sig: SignatureId,
    ) -> CallResult {
        let args_span = self.graph.value_pool.push_slice(args);

        let call_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::CallIndirect {
                state,
                callee,
                sig,
                args: args_span,
            },
        });
        let out_state = State(call_val);

        let first_res = ValueId(self.graph.value_kinds.len() as u32);
        let ret = self
            .module_tables
            .types
            .get_fn(self.module_tables.signatures.get(sig).func_type)
            .ret;
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
