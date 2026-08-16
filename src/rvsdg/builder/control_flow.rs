use color_eyre::eyre::eyre;

use crate::rvsdg::{
    AliasClassId, FuncId, MatchArm, RegionId, State, StateKind, Value, ValueId, ValueKind,
    func::SignatureId,
    types::{ScalarType, TypeRef, VOID},
};

use super::{LoopResult, NodeResults, RegionBuilder};

/// A gamma alternative's body: emits into its own fresh region and
/// returns the arm's result values.
pub type GammaBranch<'c> =
    dyn for<'g> Fn(&mut RegionBuilder<'g>) -> color_eyre::Result<Vec<ValueId>> + 'c;

impl<'a> RegionBuilder<'a> {
    /// N-way conditional branch. Condition value selects which region executes:
    /// 0 -> first branch, 1 -> second, etc. All branches must return the same
    /// number and types of values.
    #[inline]
    pub fn gamma_n(
        &mut self,
        condition: ValueId,
        inputs: &[ValueId],
        branches: &[&GammaBranch<'_>],
    ) -> color_eyre::Result<NodeResults> {
        debug_assert!(branches.len() >= 2, "gamma requires at least 2 branches");
        let mut branch_regions: Vec<RegionId> = Vec::with_capacity(branches.len());
        let mut result_count: Option<u16> = None;

        let param_types: Vec<TypeRef> = inputs
            .iter()
            .map(|&id| *self.graph.get_value_type(id))
            .collect();

        // Arms chain from the parent's currents; pending parent reads
        // stay pending and flatten into the gamma's own memory state
        // input at assembly, ordering the construct behind them.
        let entry_seeds = self.graph.entry_seeds(self.region_id);
        for branch in branches {
            let mut rb = RegionBuilder::new_with_params(
                self.graph,
                self.module_tables,
                entry_seeds,
                &param_types,
            );
            let values = branch(&mut rb)?;
            let count = values.len() as u16;
            match result_count {
                None => result_count = Some(count),
                Some(expected) => debug_assert_eq!(
                    count, expected,
                    "all gamma branches must return the same number of values"
                ),
            }
            rb.graph.seal_region(rb.region_id, &values);
            branch_regions.push(rb.region_id);
        }

        let result_count =
            result_count.ok_or_else(|| eyre!("gamma requires at least 2 branches"))?;
        Ok(self.finish_gamma(condition, inputs, &branch_regions, result_count))
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
        inputs: &[ValueId],
        branch_regions: &[RegionId],
        result_count: u16,
    ) -> NodeResults {
        debug_assert!(
            branch_regions.len() >= 2,
            "gamma requires at least 2 branches"
        );
        let inputs_span = self.graph.value_pool.push_slice(inputs);
        let regions = self.graph.region_pool.push_slice(branch_regions);

        // The construct's chain inputs are its subregions' entry tails,
        // recorded at seeding; nothing runs in the parent between the
        // seeding and this assembly.
        let region = self.region_id;
        self.graph
            .debug_assert_seeds_are_current(region, branch_regions[0]);
        let gamma_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Gamma {
                condition,
                inputs: inputs_span,
                regions,
            },
        });
        for &arm in branch_regions {
            self.graph.get_region_mut(arm).owner = gamma_val;
        }

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
        self.add_state_projections(gamma_val, result_count);

        NodeResults {
            first_result,
            result_count,
        }
    }

    /// The construct's per-chain outputs: state projections directly
    /// after the data projections, paired positionally with its
    /// subregions' exit tails ([memory, io]). The parent's chains
    /// continue on them.
    fn add_state_projections(&mut self, construct: ValueId, result_count: u16) {
        let memory = self.add_value(Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::Project {
                call: construct,
                index: result_count,
            },
        });
        let io = self.add_value(Value {
            ty: TypeRef::State(StateKind::InputOutput),
            kind: ValueKind::Project {
                call: construct,
                index: result_count + 1,
            },
        });
        self.graph
            .state_construct_outputs(self.region_id, State(memory), State(io));
    }

    /// Assemble a theta node from a body region that was built
    /// incrementally (see `finish_gamma`). The region's parameters must be
    /// aligned to `loop_vars` positionally and its results (the
    /// next-iteration values) set; `condition` is the repetition predicate,
    /// a value inside the body region (alternative 1 repeats).
    pub fn finish_theta(
        &mut self,
        loop_vars: &[ValueId],
        body_region: RegionId,
        condition: ValueId,
    ) -> NodeResults {
        let loop_span = self.graph.value_pool.push_slice(loop_vars);
        let result_count = loop_vars.len() as u16;

        // Chain inputs live on the body's entry tail; see finish_gamma.
        let region = self.region_id;
        self.graph
            .debug_assert_seeds_are_current(region, body_region);
        let theta_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Theta {
                loop_vars: loop_span,
                condition,
                region_id: body_region,
            },
        });
        self.graph.get_region_mut(body_region).owner = theta_val;

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
        self.add_state_projections(theta_val, result_count);

        NodeResults {
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
        inputs: &[ValueId],
        true_branch: impl Fn(&mut RegionBuilder) -> color_eyre::Result<Vec<ValueId>>,
        false_branch: impl Fn(&mut RegionBuilder) -> color_eyre::Result<Vec<ValueId>>,
    ) -> color_eyre::Result<NodeResults> {
        self.gamma_n(condition, inputs, &[&true_branch, &false_branch])
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
        loop_vars: &[ValueId],
        loop_body: impl FnOnce(&mut RegionBuilder) -> color_eyre::Result<LoopResult>,
    ) -> color_eyre::Result<NodeResults> {
        let param_types: Vec<TypeRef> = loop_vars
            .iter()
            .map(|id| *self.graph.get_value_type(*id))
            .collect();

        // Same entry seeding as gamma_n arms: the body chains from the
        // parent's currents.
        let entry_seeds = self.graph.entry_seeds(self.region_id);
        let (region, condition) = {
            let mut rb = RegionBuilder::new_with_params(
                self.graph,
                self.module_tables,
                entry_seeds,
                &param_types,
            );
            let res = loop_body(&mut rb)?;
            debug_assert_eq!(
                res.next_vars.len(),
                loop_vars.len(),
                "theta body must return the same number of loop vars"
            );
            rb.graph.seal_region(rb.region_id, &res.next_vars);
            (rb.region_id, res.condition)
        };

        Ok(self.finish_theta(loop_vars, region, condition))
    }

    /// Call a known function with an ABI signature derived from its
    /// declaration. Exact for hand-built graphs and any non-variadic
    /// call; a parser lowering real call sites should use
    /// [`call_with_signature`](Self::call_with_signature) with the site's
    /// own signature instead, because a variadic call site carries ABI
    /// attributes for actual arguments the declaration knows nothing
    /// about.
    #[inline]
    pub fn call(&mut self, fn_id: FuncId, args: &[ValueId]) -> NodeResults {
        let sig =
            self.module_tables.get_function(fn_id).declared_sig.expect(
                "multi-return function has no declaration signature; use call_with_signature",
            );
        self.call_with_signature(fn_id, args, sig)
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
        args: &[ValueId],
        sig: SignatureId,
    ) -> NodeResults {
        let args_span = self.graph.value_pool.push_slice(args);

        // Calls consume both chains (the callee may touch memory and
        // perform io); the call value is the output state of both.
        let region = self.region_id;
        let io_state = self.graph.state_io_current(region);
        let out_state = self.graph.state_write(region, |state| Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Call {
                state,
                io_state,
                fn_id,
                sig,
                args: args_span,
            },
        });
        let call_val = out_state.0;
        self.graph.state_io(region, call_val);

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

        NodeResults {
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
        args: &[ValueId],
        sig: SignatureId,
    ) -> NodeResults {
        let args_span = self.graph.value_pool.push_slice(args);

        let region = self.region_id;
        let io_state = self.graph.state_io_current(region);
        let out_state = self.graph.state_write(region, |state| Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::CallIndirect {
                state,
                io_state,
                callee,
                sig,
                args: args_span,
            },
        });
        let call_val = out_state.0;
        self.graph.state_io(region, call_val);

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

        NodeResults {
            first_result: first_res,
            result_count,
        }
    }
}
