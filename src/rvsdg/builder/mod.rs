mod aggregate;
mod arithmetic;
mod control_flow;
mod memory;

use crate::rvsdg::{
    FuncId, RegionId, State, Value, ValueId, ValueKind, function_graph::FunctionGraph,
    module_tables::ModuleTables, types::TypeRef,
};

/// Passed to a branch closure -- represents being inside a gamma branch
#[derive(Debug)]
pub struct RegionBuilder<'a> {
    pub region_id: RegionId,
    pub graph: &'a mut FunctionGraph,
    pub module_tables: &'a mut ModuleTables,
}

impl<'a> RegionBuilder<'a> {
    /// A builder over an already-created region, for incremental construct
    /// assembly: the region is created first (`add_region`), emitted into,
    /// and the enclosing gamma/theta node built afterwards.
    pub fn over(
        graph: &'a mut FunctionGraph,
        module_tables: &'a mut ModuleTables,
        region_id: RegionId,
    ) -> Self {
        Self {
            region_id,
            module_tables,
            graph,
        }
    }

    pub fn new_with_params(
        graph: &'a mut FunctionGraph,
        module_tables: &'a mut ModuleTables,
        entry_state: State,
        param_types: &[TypeRef],
    ) -> Self {
        let region = graph.create_region(entry_state);
        for &ty in param_types {
            graph.append_region_param(region, ty);
        }
        Self {
            region_id: region,
            module_tables,
            graph,
        }
    }

    pub fn new_from_func(
        graph: &'a mut FunctionGraph,
        module_tables: &'a mut ModuleTables,
        func_id: FuncId,
    ) -> Self {
        // The entry state is not known until the params exist; created
        // INVALID and stamped below.
        let region = graph.create_region(State::INVALID);
        let function = module_tables.get_function(func_id);
        for param in &function.params {
            graph.append_region_param(region, param.ty);
        }

        // Add the state parameter after all value params. It is NOT in
        // the params segment (callers pass no state argument); its index
        // is one past the list, and dead node elimination's renumbering
        // preserves that convention.
        let entry_state = State(ValueId(graph.value_kinds.len() as u32));
        graph.value_kinds.push(ValueKind::RegionParam {
            index: function.params.len() as u32,
            ty: TypeRef::State,
            region,
        });
        graph.value_types.push(TypeRef::State);
        graph.regions[region.0 as usize].entry_state = entry_state;

        Self {
            region_id: region,
            module_tables,
            graph,
        }
    }

    #[inline]
    pub fn region_id(&self) -> RegionId {
        self.region_id
    }

    #[inline]
    pub fn param(&self, index: u32) -> ValueId {
        self.graph.region_params(self.region_id)[index as usize]
    }

    #[inline]
    pub fn add_region(&mut self, state: State) -> RegionId {
        self.graph.create_region(state)
    }

    #[inline(always)]
    pub(crate) fn add_value(&mut self, data: Value) -> ValueId {
        let id = ValueId(self.graph.value_kinds.len() as u32);
        self.graph.value_kinds.push(data.kind);
        self.graph.value_types.push(data.ty);
        self.graph.push_region_node(self.region_id, id);
        id
    }
}

// -- Result types ------------------------------------------------

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

// TODO: BranchResult and LoopResult each allocate a Vec<ValueId> per closure
// invocation. These are short-lived (created in the closure, consumed
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
pub struct LoopResult {
    /// If true, loop continues; if false, loop exits
    pub condition: ValueId,
    pub next_state: State,
    pub next_vars: Vec<ValueId>,
}

#[cfg(test)]
mod test {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, Linkage, RVSDGMod,
        func::FnResult,
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

        let mut rvsdg_mod = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg_mod.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg_mod
            .define_fn(main_fn, |rb, state| {
                let a = rb.const_i32(5);
                let b = rb.const_i32(3);
                let c = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
                Ok(FnResult {
                    state,
                    values: vec![c],
                })
            })
            .unwrap();
    }

    #[test]
    fn test_comparison() {
        // bool check(int x, int y) {
        //   return x < y;
        // }

        let mut rvsdg_mod = RVSDGMod::new_host(String::from("test"));
        let check_fn = rvsdg_mod.declare_fn(
            String::from("check"),
            &[I32, I32],
            &[BOOL],
            Linkage::Internal,
        );
        rvsdg_mod
            .define_fn(check_fn, |rb, state| {
                let x = rb.param(0);
                let y = rb.param(1);
                let result = rb.icmp(ICmpPred::SignedLt, x, y);
                Ok(FnResult {
                    state,
                    values: vec![result],
                })
            })
            .unwrap();
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

        let mut rvsdg_mod = RVSDGMod::new_host(String::from("test"));

        let check_fn = rvsdg_mod.declare_fn(
            String::from("check"),
            &[I32, I32],
            &[BOOL],
            Linkage::Internal,
        );
        rvsdg_mod
            .define_fn(check_fn, |rb, state| {
                let x = rb.param(0);
                let y = rb.param(1);
                let result = rb.icmp(ICmpPred::SignedLt, x, y);
                Ok(FnResult {
                    state,
                    values: vec![result],
                })
            })
            .unwrap();

        let main_fn = rvsdg_mod.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg_mod
            .define_fn(main_fn, |rb, entry_state| {
                let a = rb.const_i32(5);
                let b = rb.const_i32(3);
                let call_res = rb.call(check_fn, entry_state, &[a, b]);
                let c = call_res.result(0);
                Ok(FnResult {
                    state: call_res.state,
                    values: vec![c],
                })
            })
            .unwrap();
    }
}
