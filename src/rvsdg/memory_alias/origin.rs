use crate::rvsdg::{
    FuncId, GlobalId, ValueId,
    constant::{ConstId, ConstantKind, ConstantPool},
    function_graph::FunctionGraph,
    types::TypeRef,
};

const _: () = assert!(size_of::<MemoryOrigin>() == 8);

/// Information about where the memory came from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrigin {
    /// Not a pointer or untracked
    None,

    /// Stack allocated
    Alloca(ValueId),

    /// From this functions nth param
    Param(u32),

    /// From a global address
    Global(GlobalId),

    /// Same origin as another value, this case should only exist during construction.
    Derived(ValueId),

    /// A pointer returned by this call. The origin depends on the
    /// callee's return provenance (interprocedural), so this is the one
    /// tag that outlives resolution: opaque within the function,
    /// resolved by classing through the callee's summary.
    CallResult(ValueId),

    /// The address of a function. Not data memory -- it exists so the
    /// address SURVIVES derivation: `cond ? &f : &g` widens through a
    /// join, and the discarded named origins must still reach
    /// address_taken_functions, which a kind check on the escaping
    /// value alone can never see.
    Func(FuncId),

    /// Loaded, laundered, or a join of disagreeing origins
    Unknown,
}

impl MemoryOrigin {
    /// The origin of a value that came out of somewhere origins cannot
    /// follow -- memory (loads, atomics) or an aggregate/vector
    /// (extractions): Unknown for pointers, None otherwise.
    #[inline]
    pub(crate) fn unknown_if_ptr(ty: TypeRef) -> Self {
        if matches!(ty, TypeRef::Ptr(_)) {
            MemoryOrigin::Unknown
        } else {
            MemoryOrigin::None
        }
    }

    /// Bring an origin through a dead-node-elimination compaction: a
    /// payload VALUE id (Alloca/Derived/CallResult) follows the value
    /// mapper (Func carries a FuncId, which DNE never renumbers);
    /// a DEAD payload widens to Unknown, which is sound -- a dead
    /// origin has no surviving accesses to conflict with.
    #[inline]
    pub(crate) fn remap(self, value_mapper: &[u32]) -> Self {
        let map = |v: ValueId| {
            let new = value_mapper[v.0 as usize];
            (new != u32::MAX).then_some(ValueId(new))
        };
        match self {
            MemoryOrigin::Alloca(v) => map(v).map_or(MemoryOrigin::Unknown, MemoryOrigin::Alloca),
            MemoryOrigin::Derived(v) => map(v).map_or(MemoryOrigin::Unknown, MemoryOrigin::Derived),
            MemoryOrigin::CallResult(v) => {
                map(v).map_or(MemoryOrigin::Unknown, MemoryOrigin::CallResult)
            }
            MemoryOrigin::None
            | MemoryOrigin::Param(_)
            | MemoryOrigin::Global(_)
            | MemoryOrigin::Func(_)
            | MemoryOrigin::Unknown => self,
        }
    }

    /// The origin of a pointer-typed pool constant, resolved to its
    /// base: the parser reaches most global addresses through the
    /// constant pool (`&g[3]` is a GEP-of-GlobalAddr entry), not
    /// through intern_global_ref, and missing this would leave stored
    /// global addresses looking untracked. Anything without a single
    /// resolvable base is Unknown, which is always sound (class 0).
    pub(crate) fn of_constant(pool: &ConstantPool, id: ConstId) -> Self {
        match &pool.get(id).kind {
            ConstantKind::GlobalAddr(global) => MemoryOrigin::Global(*global),
            ConstantKind::FuncAddr(function) => MemoryOrigin::Func(*function),
            ConstantKind::GetElementPointer { base, .. } => Self::of_constant(pool, *base),
            ConstantKind::Cast { operand, .. } => Self::of_constant(pool, *operand),
            // Null, undef, aggregates, strings: no single base.
            _ => MemoryOrigin::Unknown,
        }
    }
}

impl FunctionGraph {
    #[inline]
    pub fn get_memory_origin(&self, value: ValueId) -> MemoryOrigin {
        self.memory_origins[value.0 as usize]
    }

    #[inline]
    pub fn set_memory_origin(&mut self, value: ValueId, origin: MemoryOrigin) {
        self.memory_origins[value.0 as usize] = origin;
    }
}

#[cfg(test)]
mod test {
    use super::MemoryOrigin;
    use crate::rvsdg::{
        ArithFlags, BinaryOp, Linkage, RVSDGMod, ValueId,
        types::{I32, PtrType, TypeRef},
    };

    /// The values the body closure smuggles out for assertions.
    struct BuiltIds {
        param_arr: ValueId,
        stack_arr: ValueId,
        param_first: ValueId,
        stack_first: ValueId,
        val: ValueId,
        stack_add_1: ValueId,
        param_add_1: ValueId,
    }

    // int* f(int *param_arr) {
    //      int stack_arr[] = {0, 1, 2, 3}; // alloca
    //      int param_first = param_arr[0]; // none
    //      int stack_first = stack_arr[0]; // none
    //      stack_arr[1] = param_first + stack_first; // alloca
    //      param_arr[1] = param_first + stack_first; // param
    //      return &param_arr[1]; // param
    // }
    #[test]
    fn basic_mem_origin_builder() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let i32_ptr_id = m.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        });
        let i32_ptr_ty = TypeRef::Ptr(i32_ptr_id);
        let f = m.declare_fn(
            String::from("f"),
            &[i32_ptr_ty; 1],
            &[i32_ptr_ty; 1],
            Linkage::Internal,
        );
        let mut ids = None;
        m.define_fn(f, |rb| {
            let param_arr = rb.param(0);
            let four = rb.const_i32(4);
            let stack_arr = rb.alloca(I32, four, i32_ptr_ty, None);
            let param_first = rb.load(param_arr, I32, None, false);
            let stack_first = rb.load(stack_arr, I32, None, false);
            let val = rb.binary(
                BinaryOp::Add,
                ArithFlags::default(),
                param_first,
                stack_first,
                I32,
            );
            let one = rb.const_i32(1);
            // ptr_offset's 4th argument is the RESULT type: &stack_arr[1]
            // is a pointer, not an i32.
            let stack_add_1 = rb.ptr_offset(stack_arr, I32, &[one], i32_ptr_ty, true);
            rb.store(stack_add_1, val, None, false);
            let param_add_1 = rb.ptr_offset(param_arr, I32, &[one], i32_ptr_ty, true);
            rb.store(param_add_1, val, None, false);
            // Event smoke checks, observable only here: resolve_facts
            // drains the scratch at define_fn's end. Two loads + two
            // stores; no escapes (the stored values are ints, and the
            // pointer RETURN is recorded by define_fn after this
            // closure runs).
            assert_eq!(rb.graph.mem_facts.access_events.len(), 4);
            assert!(rb.graph.mem_facts.escape_events.is_empty());
            assert!(rb.graph.mem_facts.join_events.is_empty());
            assert!(rb.graph.mem_facts.call_sites.is_empty());
            ids = Some(BuiltIds {
                param_arr,
                stack_arr,
                param_first,
                stack_first,
                val,
                stack_add_1,
                param_add_1,
            });
            Ok(vec![param_add_1])
        })
        .unwrap();
        let ids = ids.unwrap();

        let errs = m.verify();
        assert!(errs.is_empty(), "graph does not verify: {errs:?}");

        let g = m.graphs[f.0 as usize].as_ref().unwrap();

        // Roots: the parameter and the alloca's pointer projection. The
        // alloca origin must name the alloca NODE, not the projection --
        // recover the node through the projection's kind so an
        // off-by-one (projection id instead of node id) cannot pass.
        assert_eq!(g.get_memory_origin(ids.param_arr), MemoryOrigin::Param(0));
        let crate::rvsdg::ValueKind::Project {
            call: alloca_node, ..
        } = *g.get_value_kind(ids.stack_arr)
        else {
            panic!("alloca builder must return the pointer projection");
        };
        assert_eq!(
            g.get_memory_origin(ids.stack_arr),
            MemoryOrigin::Alloca(alloca_node)
        );

        // Pointer arithmetic links to its base during construction
        // (never a copy -- the base may widen later; see the one rule);
        // define_fn's resolve_facts has since COMPRESSED the links to
        // their final origins.
        assert_eq!(
            g.get_memory_origin(ids.stack_add_1),
            MemoryOrigin::Alloca(alloca_node)
        );
        assert_eq!(g.get_memory_origin(ids.param_add_1), MemoryOrigin::Param(0));

        // Origins describe POINTERS only. The loaded ints and their sum
        // carry nothing, no matter which memory they were loaded from.
        for v in [ids.param_first, ids.stack_first, ids.val] {
            assert_eq!(g.get_memory_origin(v), MemoryOrigin::None);
        }

        // Every value has a side-table entry: the arrays cannot drift.
        assert_eq!(g.memory_origins.len(), g.value_kinds.len());

        // And the facts define_fn resolved: returning &param_arr[1]
        // retains parameter 0 and names it as the return's origin;
        // param traffic is ReadAndWrite on entry 0; nothing escaped.
        use crate::rvsdg::func::MemReadWrite;
        use crate::rvsdg::memory_alias::LocalReturn;
        let facts = &m.facts[f.0 as usize];
        assert_eq!(facts.local_return, LocalReturn::Param(0));
        assert!(facts.local_captured.is_retained(0));
        assert_eq!(facts.local_param_effects.get(0), MemReadWrite::ReadAndWrite);
        assert!(facts.escaped_origins.is_empty());
        assert_eq!(facts.local_other, MemReadWrite::None);
        assert!(facts.calls.is_empty());
    }
}
