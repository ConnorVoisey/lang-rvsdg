mod aggregate;
mod arithmetic;
mod control_flow;
mod memory;

use crate::rvsdg::{
    AliasClassId, FuncId, RegionId, State, StateKind, Value, ValueId, ValueKind,
    function_graph::FunctionGraph,
    module_tables::ModuleTables,
    state::{MemoryStates, StateGroup},
    types::TypeRef,
};

/// Emission handle for one open region (the function body or a
/// gamma/theta subregion): borrows the graph and module tables and
/// appends values into `region_id`.
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
        entry: StateGroup,
        param_types: &[TypeRef],
    ) -> Self {
        let region = graph.create_region(entry);
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
        // The entry states are not known until the params exist; created
        // INVALID and stamped below.
        let invalid = StateGroup {
            memory: MemoryStates {
                read: State::INVALID,
                write: State::INVALID,
            },
            io: State::INVALID,
        };
        let region = graph.create_region(invalid);
        let function = module_tables.get_function(func_id);
        for param in &function.params {
            graph.append_region_param(region, param.ty);
        }

        // Add one state parameter per chain after all value params, in
        // tail order [memory, io]. They are NOT in the params segment
        // (callers pass no state arguments); seal writes them into the
        // state-params tail from the scratch entry group.
        let value_param_count = function.params.len() as u32;
        let state_param = |graph: &mut FunctionGraph, index_offset: u32, ty: TypeRef| {
            let id = ValueId(graph.value_kinds.len() as u32);
            graph.value_kinds.push(ValueKind::RegionParam {
                index: value_param_count + index_offset,
                ty,
                region,
            });
            graph.value_types.push(ty);
            State(id)
        };
        let memory = state_param(
            graph,
            0,
            TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
        );
        let io = state_param(graph, 1, TypeRef::State(StateKind::InputOutput));
        graph.set_scratch_entry_state(
            region,
            StateGroup {
                memory: MemoryStates {
                    read: memory,
                    write: memory,
                },
                io,
            },
        );

        Self {
            region_id: region,
            module_tables,
            graph,
        }
    }

    #[inline]
    pub fn param(&self, index: u32) -> ValueId {
        self.graph.region_params(self.region_id)[index as usize]
    }

    /// Create a subregion for incremental construct assembly, seeded
    /// with this region's current states as its entry (pending parent
    /// reads stay pending and flatten into the construct's memory state
    /// input at assembly).
    #[inline]
    pub fn add_region(&mut self) -> RegionId {
        let seeds = self.graph.entry_seeds(self.region_id);
        self.graph.create_region(seeds)
    }

    #[inline(always)]
    pub(crate) fn add_value(&mut self, data: Value) -> ValueId {
        self.graph.add_region_value(self.region_id, data)
    }
}

// -- Result types ------------------------------------------------
//
// State threading is internal to the builder, so results carry data
// values only. Multi-value results use named structs, never tuples.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompareAndSwapResult {
    /// The compare-and-swap node itself. The `{old value, success}` pair
    /// has no aggregate value in the RVSDG; a frontend lowering an
    /// aggregate-consuming instruction (LLVM's extractvalue) binds the
    /// node and routes field reads to its projections.
    pub node: ValueId,
    pub old_value: ValueId,
    pub success: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverflowResult {
    pub value: ValueId,
    pub overflow: ValueId,
}

/// The contiguous data result projections of a multi-output node
/// (gamma, theta, call): the adjacency `projection_of` checks, handed
/// back by the builders that create such nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeResults {
    pub first_result: ValueId,
    pub result_count: u16,
}

impl NodeResults {
    pub fn result(&self, index: u16) -> ValueId {
        debug_assert!(index < self.result_count);
        ValueId(self.first_result.0 + index as u32)
    }
}

// TODO: LoopResult allocates a Vec<ValueId> per closure invocation.
// These are short-lived (created in the closure, consumed immediately
// by the builder). For typical loops carrying 1-3 vars this is likely
// fine, but profile real-world code to determine if SmallVec<[ValueId; 4]>
// would be worthwhile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopResult {
    /// If true, loop continues; if false, loop exits
    pub condition: ValueId,
    pub next_vars: Vec<ValueId>,
}

#[cfg(test)]
mod test {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, Linkage, RVSDGMod,
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
            .define_fn(main_fn, |rb| {
                let a = rb.const_i32(5);
                let b = rb.const_i32(3);
                let c = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
                Ok(vec![c])
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
            .define_fn(check_fn, |rb| {
                let x = rb.param(0);
                let y = rb.param(1);
                let result = rb.icmp(ICmpPred::SignedLt, x, y);
                Ok(vec![result])
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
            .define_fn(check_fn, |rb| {
                let x = rb.param(0);
                let y = rb.param(1);
                let result = rb.icmp(ICmpPred::SignedLt, x, y);
                Ok(vec![result])
            })
            .unwrap();

        let main_fn = rvsdg_mod.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg_mod
            .define_fn(main_fn, |rb| {
                let a = rb.const_i32(5);
                let b = rb.const_i32(3);
                let call_res = rb.call(check_fn, &[a, b]);
                let c = call_res.result(0);
                Ok(vec![c])
            })
            .unwrap();
    }
}
