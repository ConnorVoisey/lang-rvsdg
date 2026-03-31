mod aggregate;
mod arithmetic;
mod control_flow;
mod memory;

use crate::rvsdg::{
    FuncId, RVSDGMod, Region, RegionId, State, Value, ValueId, ValueKind, ValuesSpan,
    types::TypeRef,
};

/// Passed to a branch closure — represents being inside a gamma branch
#[derive(Debug)]
pub struct RegionBuilder<'a> {
    pub region_id: RegionId,
    pub graph: &'a mut RVSDGMod,
}

impl<'a> RegionBuilder<'a> {
    pub fn new_empty(graph: &'a mut RVSDGMod, entry_state: State) -> Self {
        let region = RegionId(graph.regions.len() as u32);
        graph.regions.push(Region {
            params: ValuesSpan { start: 0, len: 0 },
            results: ValuesSpan { start: 0, len: 0 },
            entry_state,
            nodes: vec![],
        });
        Self {
            region_id: region,
            graph,
        }
    }

    pub fn new_from_func(graph: &'a mut RVSDGMod, func_id: FuncId) -> Self {
        let region = RegionId(graph.regions.len() as u32);
        let fn_params = &graph.functions[func_id.0 as usize].params;

        let param_count = fn_params.len() as u16;
        let params = {
            let val_start = graph.values.len() as u32;
            for (i, param) in fn_params.iter().enumerate() {
                graph.values.push(Value {
                    ty: param.ty,
                    kind: ValueKind::RegionParam {
                        index: i as u32,
                        ty: param.ty,
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
                index: fn_params.len() as u32,
                ty: TypeRef::State,
            },
        });

        graph.regions.push(Region {
            params,
            entry_state,
            results: ValuesSpan { start: 0, len: 0 },
            nodes: vec![],
        });
        Self {
            region_id: region,
            graph,
        }
    }

    #[inline]
    pub fn region_id(&self) -> RegionId {
        self.region_id
    }

    #[inline]
    pub fn param(&self, index: u32) -> ValueId {
        let span = self.graph.regions[self.region_id.0 as usize].params;
        debug_assert!(span.len as u32 > index);
        ValueId(span.start + index)
    }

    /// TODO: this probably needs to take in inputs and outputs
    #[inline]
    pub fn add_region(&mut self, state: State) -> RegionId {
        let id = RegionId(self.graph.regions.len() as u32);
        self.graph.regions.push(Region {
            entry_state: state,
            params: ValuesSpan { start: 0, len: 0 },
            results: ValuesSpan { start: 0, len: 0 },
            nodes: vec![],
        });
        id
    }

    #[inline]
    pub(crate) fn add_value(&mut self, data: Value) -> ValueId {
        let id = ValueId(self.graph.values.len() as u32);
        self.graph.values.push(data);
        let region = self.graph.get_region_mut(self.region_id());
        region.nodes.push(id);
        id
    }
}

// ── Result types ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadResult {
    pub state: State,
    pub value: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocaResult {
    pub state: State,
    pub ptr: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompareAndSwapResult {
    pub state: State,
    pub old_value: ValueId,
    pub success: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntrinsicResult {
    pub state: State,
    pub value: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverflowResult {
    pub state: State,
    pub value: ValueId,
    pub overflow: ValueId,
}

// TODO: BranchResult, LoopResult, and PhiBody each allocate a Vec<ValueId> per
// closure invocation. These are short-lived (created in the closure, consumed
// immediately by the builder). For typical branches returning 1-3 values this is
// likely fine, but profile real-world code to determine if SmallVec<[ValueId; 4]>
// would be worthwhile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchResult {
    pub state: State,
    pub values: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GammaResult {
    pub state: State,
    pub first_result: ValueId,
    pub result_count: u16,
}

impl GammaResult {
    pub fn result(&self, index: u16) -> ValueId {
        debug_assert!(index < self.result_count);
        ValueId(self.first_result.0 + index as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThetaResult {
    pub state: State,
    pub first_result: ValueId,
    pub result_count: u16,
}

impl ThetaResult {
    pub fn result(&self, index: u16) -> ValueId {
        debug_assert!(index < self.result_count);
        ValueId(self.first_result.0 + index as u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiBody {
    pub values: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhiResult {
    pub first_result: ValueId,
    pub result_count: u16,
}

impl PhiResult {
    pub fn result(&self, index: u16) -> ValueId {
        debug_assert!(index < self.result_count);
        ValueId(self.first_result.0 + index as u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

        let mut rvsdg_mod = RVSDGMod::new_host();
        let main_fn =
            rvsdg_mod.declare_fn(String::from("main"), &[], &[I32], FnLinkageType::Internal);
        rvsdg_mod.define_fn(main_fn, |rb, state| {
            let a = rb.const_i32(5);
            let b = rb.const_i32(3);
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

        let mut rvsdg_mod = RVSDGMod::new_host();
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

        let mut rvsdg_mod = RVSDGMod::new_host();

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
            let a = rb.const_i32(5);
            let b = rb.const_i32(3);
            let call_res = rb.call(check_fn, entry_state, &[a, b]);
            let c = call_res.result(0);
            FnResult {
                state: call_res.state,
                values: vec![c],
            }
        });
    }
}
