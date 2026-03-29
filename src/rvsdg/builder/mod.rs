use crate::rvsdg::{
    ArithFlags, BinaryOp, ConstValue, FCmpPred, FuncId, ICmpPred, RVSDGMod, Region, RegionId,
    State, Value, ValueId, ValueKind, ValuesSpan,
    func::CallResult,
    types::{BOOL, I32, I64, ScalarType, TypeRef},
};

/// Passed to a branch closure — represents being inside a gamma branch
pub struct RegionBuilder<'a> {
    pub region: RegionId,
    pub graph: &'a mut RVSDGMod,
}
impl<'a> RegionBuilder<'a> {
    pub fn new_empty(graph: &'a mut RVSDGMod, entry_state: State) -> Self {
        let region = RegionId(graph.regions.len() as u32);
        graph.regions.push(Region {
            id: region,
            params: ValuesSpan { start: 0, len: 0 },
            results: ValuesSpan { start: 0, len: 0 },
            entry_state,
            nodes: vec![],
        });
        Self { region, graph }
    }
    pub fn new_from_func(graph: &'a mut RVSDGMod, func_id: FuncId) -> Self {
        let region = RegionId(graph.regions.len() as u32);
        let param_types = &graph.functions[func_id.0 as usize].params;

        let param_count = param_types.len() as u16;
        let params = {
            let val_start = graph.values.len() as u32;
            for (i, param_ty) in param_types.iter().enumerate() {
                graph.values.push(Value {
                    ty: *param_ty,
                    kind: ValueKind::RegionParam {
                        index: i as u32,
                        ty: *param_ty,
                    },
                });
            }

            ValuesSpan {
                start: val_start,
                len: param_count,
            }
        };

        // add state node after all params, this is not recorded in the result ValuesSpan
        let entry_state = State(ValueId(graph.values.len() as u32));
        graph.values.push(Value {
            ty: TypeRef::State,
            kind: ValueKind::RegionParam {
                index: param_types.len() as u32,
                ty: TypeRef::State,
            },
        });

        graph.regions.push(Region {
            id: region,
            params,
            entry_state,
            results: ValuesSpan { start: 0, len: 0 },
            nodes: vec![],
        });
        Self { region, graph }
    }

    #[inline]
    pub fn region_id(&self) -> RegionId {
        self.region
    }

    #[inline]
    pub fn param(&self, index: u32) -> ValueId {
        let span = self.graph.regions[self.region.0 as usize].params;
        debug_assert!(span.len as u32 > index);
        ValueId(span.start + index)
    }

    /// TODO: this probably needs to take in inputs and outputs
    #[inline]
    pub fn add_region(&mut self, state: State) -> RegionId {
        let id = RegionId(self.graph.regions.len() as u32);
        self.graph.regions.push(Region {
            id,
            entry_state: state,
            params: ValuesSpan { start: 0, len: 0 },
            results: ValuesSpan { start: 0, len: 0 },
            nodes: vec![],
        });
        id
    }

    #[inline]
    pub fn add_value(&mut self, data: Value) -> ValueId {
        let id = ValueId(self.graph.values.len() as u32);
        self.graph.values.push(data);
        id
    }

    #[inline]
    pub fn add_const(&mut self, ty: TypeRef, const_val: ConstValue) -> ValueId {
        // TODO: add constant dedupe here
        self.add_value(Value {
            ty,
            kind: ValueKind::Const(const_val),
        })
    }

    #[inline]
    pub fn add_const_i32(&mut self, val: i32) -> ValueId {
        self.add_const(I32, ConstValue::Int(val as i64))
    }

    #[inline]
    pub fn add_const_i64(&mut self, val: i64) -> ValueId {
        self.add_const(I64, ConstValue::Int(val))
    }

    #[inline]
    pub fn binary(
        &mut self,
        op: BinaryOp,
        flags: ArithFlags,
        left: ValueId,
        right: ValueId,
        ret_type: TypeRef,
    ) -> ValueId {
        self.add_value(Value {
            ty: ret_type,
            kind: ValueKind::Binary {
                op,
                flags,
                left,
                right,
            },
        })
    }

    #[inline]
    pub fn icmp(&mut self, pred: ICmpPred, left: ValueId, right: ValueId) -> ValueId {
        self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::ICmp { pred, left, right },
        })
    }

    #[inline]
    pub fn fcmp(&mut self, pred: FCmpPred, left: ValueId, right: ValueId) -> ValueId {
        self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::FCmp { pred, left, right },
        })
    }

    #[inline]
    pub fn gamma(
        &mut self,
        condition: ValueId,
        state: State,
        inputs: Vec<ValueId>,
        true_branch: impl FnOnce(&mut RegionBuilder) -> BranchResult,
        false_branch: impl FnOnce(&mut RegionBuilder) -> BranchResult,
    ) {
        let _true_region = self.add_region(state);
        let _false_region = self.add_region(state);

        // TODO: construct RegionBuilders for branches and wire up gamma node
        todo!()
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
        let ret_types = &self.graph.functions[fn_id.0 as usize].return_types;
        let result_count = ret_types.len() as u16;
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
}

pub struct BranchResult {
    pub state: State,
    pub values: Vec<ValueId>,
}

pub struct LoopResult {
    /// If true, loop continues; if false, loop exits
    pub condition: ValueId,
    pub next_state: State,
    pub next_vars: Vec<ValueId>,
}

#[cfg(test)]
mod test {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, RVSDGMod,
        func::{FnLinkageType, FnResult},
        types::{BOOL, I32},
    };

    #[test]
    fn test_example() {
        // int main() {
        //   int a = 5;
        //   int b = 3;
        //   int c = a + b;
        //   return c;
        // }

        let mut rvsdg_mod = RVSDGMod::new();
        let main_fn =
            rvsdg_mod.declare_fn(String::from("main"), &[], &[I32], FnLinkageType::Internal);
        rvsdg_mod.define_fn(main_fn, |rb, state| {
            let a = rb.add_const_i32(5);
            let b = rb.add_const_i32(3);
            let c = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state,
                values: vec![c],
            }
        });
    }

    #[test]
    fn test_comparison() {
        // bool check(int x, int y) {
        //   return x < y;
        // }

        let mut rvsdg_mod = RVSDGMod::new();
        let check_fn = rvsdg_mod.declare_fn(
            String::from("check"),
            &[I32, I32],
            &[BOOL],
            FnLinkageType::Internal,
        );
        rvsdg_mod.define_fn(check_fn, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let result = rb.icmp(ICmpPred::SignedLt, x, y);
            FnResult {
                state,
                values: vec![result],
            }
        });
    }

    #[test]
    fn test_call() {
        // bool check(int x, int y) {
        //   return x < y;
        // }
        //
        // bool main() {
        //   int a = 5;
        //   int b = 3;
        //   bool c = check(a, b)
        //   return c;
        // }

        let mut rvsdg_mod = RVSDGMod::new();

        let check_fn = rvsdg_mod.declare_fn(
            String::from("check"),
            &[I32, I32],
            &[BOOL],
            FnLinkageType::Internal,
        );
        rvsdg_mod.define_fn(check_fn, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let result = rb.icmp(ICmpPred::SignedLt, x, y);
            FnResult {
                state,
                values: vec![result],
            }
        });

        let main_fn =
            rvsdg_mod.declare_fn(String::from("main"), &[], &[I32], FnLinkageType::Internal);
        rvsdg_mod.define_fn(main_fn, |rb, entry_state| {
            let a = rb.add_const_i32(5);
            let b = rb.add_const_i32(3);
            let call_res = rb.call(check_fn, entry_state, &[a, b]);
            let c = call_res.result(0);
            FnResult {
                state: call_res.state,
                values: vec![c],
            }
        });
    }
}
