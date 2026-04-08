use crate::rvsdg::{
    FuncId, RegionId, State, Value, ValueId, ValueKind,
    func::CallResult,
    types::{ScalarType, TypeRef},
};

use super::{
    BranchResult, GammaResult, LoopResult, PhiBody, PhiResult, RegionBuilder, ThetaResult,
};

impl<'a> RegionBuilder<'a> {
    /// N-way conditional branch. Condition value selects which region executes:
    /// 0 → first branch, 1 → second, etc. All branches must return the same
    /// number and types of values.
    // TODO: `inputs` requires the caller to collect all live values into a slice,
    // which may cause a Vec allocation for complex branches with many live variables
    // (20-50+ in real programs). Profile real-world code to determine if an
    // incremental builder API (gamma_begin/add_input/build) would be worthwhile.
    #[inline]
    pub fn gamma_n(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: &[ValueId],
        branches: &[&dyn Fn(&mut RegionBuilder) -> BranchResult],
    ) -> GammaResult {
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
            let res = branch(&mut rb);
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

        let result_count = result_count.unwrap();
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

        GammaResult {
            state: out_state,
            first_result,
            result_count,
        }
    }

    /// Two-way if/else convenience. Condition is a bool: true → first branch,
    /// false → second branch. See `gamma_n` for the inputs allocation note.
    #[inline]
    pub fn gamma(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: &[ValueId],
        true_branch: impl FnOnce(&mut RegionBuilder) -> BranchResult,
        false_branch: impl FnOnce(&mut RegionBuilder) -> BranchResult,
    ) -> GammaResult {
        let inputs_span = self.graph.value_pool.push_slice(inputs);
        let param_types: Vec<TypeRef> = inputs
            .iter()
            .map(|&id| self.graph.values[id.0 as usize].ty)
            .collect();

        let (true_region, result_count) = {
            let mut rb = RegionBuilder::new_with_params(self.graph, state, &param_types);
            let res = true_branch(&mut rb);
            let count = res.values.len() as u16;
            let results = rb.graph.value_pool.push_slice(&res.values);
            rb.graph.regions[rb.region_id.0 as usize].results = results;
            (rb.region_id, count)
        };

        let false_region = {
            let mut rb = RegionBuilder::new_with_params(self.graph, state, &param_types);
            let res = false_branch(&mut rb);
            debug_assert_eq!(
                res.values.len() as u16,
                result_count,
                "gamma branches must return the same number of values"
            );
            let results = rb.graph.value_pool.push_slice(&res.values);
            rb.graph.regions[rb.region_id.0 as usize].results = results;
            rb.region_id
        };

        let regions = self
            .graph
            .region_pool
            .push_slice(&[true_region, false_region]);

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
        let true_results = self.graph.regions[true_region.0 as usize].results;
        for i in 0..result_count {
            let ty = self.graph.values
                [self.graph.value_pool.get(true_results)[i as usize].0 as usize]
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

    #[inline]
    pub fn theta(
        &mut self,
        state: State,
        loop_vars: &[ValueId],
        loop_body: impl FnOnce(&mut RegionBuilder) -> LoopResult,
    ) -> ThetaResult {
        let loop_span = self.graph.value_pool.push_slice(loop_vars);
        let result_count = loop_vars.len() as u16;

        let param_types: Vec<TypeRef> = loop_vars
            .iter()
            .map(|&id| self.graph.values[id.0 as usize].ty)
            .collect();

        let (region, condition) = {
            let mut rb = RegionBuilder::new_with_params(self.graph, state, &param_types);
            let res = loop_body(&mut rb);
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

        ThetaResult {
            state: out_state,
            first_result,
            result_count,
        }
    }

    /// Build a phi node for mutually recursive function definitions.
    ///
    /// `rv_count` is the number of recursion variables (one per mutually recursive function).
    /// The closure receives a `RegionBuilder` whose first `rv_count` params are the recursion
    /// variables — handles the body can use to refer to the functions being defined.
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

    /// Call through a function pointer. The caller must provide the return types
    /// since they can't be looked up from the function table.
    #[inline]
    pub fn call_indirect(
        &mut self,
        callee: ValueId,
        state: State,
        args: &[ValueId],
        return_types: &[TypeRef],
    ) -> CallResult {
        let args_span = self.graph.value_pool.push_slice(args);

        let call_val = self.add_value(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::CallIndirect {
                state,
                callee,
                args: args_span,
            },
        });
        let out_state = State(call_val);

        let first_res = ValueId(self.graph.values.len() as u32);
        let result_count = return_types.len() as u16;
        for i in 0..result_count {
            self.add_value(Value {
                ty: return_types[i as usize],
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
