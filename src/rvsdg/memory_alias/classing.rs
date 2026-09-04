//! Alias classing: decide which alias class every memory origin
//! belongs to. Stage one recognises two kinds of proof -- class 0, the
//! escaped world, and one class per PRIVATE alloca (nothing outside the
//! function can name it). The split pass consumes the assignment and
//! re-threads chains to match; this pass only decides.
//!
//! Classing reads FunctionFacts alone: the alloca inventory, the fold
//! sets, and the escape sets were all recorded while construction and
//! event classification were already standing at the relevant values,
//! so no pass here walks the graph or allocates scratch.

use crate::rvsdg::{
    AliasClassId, ValueId,
    memory_alias::{FunctionFacts, origin::MemoryOrigin},
};

/// What classing needs to know about the world outside the function.
/// The split never looks past this seam: the facts-only tier answers
/// from per-function facts alone, and the summaries tier later swaps in
/// an impl that consults callee summaries -- replacing the INPUTS, not
/// the classing pass.
pub(crate) trait ClassingInputs {
    /// True when something outside the function may reach this origin
    /// (an alloca node id): such an origin is class 0 no matter how it
    /// is used locally.
    fn origin_escaped(&self, origin: ValueId) -> bool;
}

/// The facts-only tier: escaped = the hard escapes UNION every alloca
/// handed to a call -- with no callee knowledge, every call retains
/// what it is handed. Both sets are sorted facts fields, so the tier
/// borrows them and allocates nothing.
pub(crate) struct FactsClassingInputs<'facts> {
    facts: &'facts FunctionFacts,
}

impl<'facts> FactsClassingInputs<'facts> {
    pub(crate) fn new(facts: &'facts FunctionFacts) -> Self {
        FactsClassingInputs { facts }
    }
}

impl ClassingInputs for FactsClassingInputs<'_> {
    fn origin_escaped(&self, origin: ValueId) -> bool {
        contains_id(&self.facts.escaped_origins, origin)
            || contains_id(&self.facts.call_retained, origin)
    }
}

/// Membership in a sorted-by-id list.
fn contains_id(list: &[ValueId], target: ValueId) -> bool {
    list.binary_search_by_key(&target.0, |value| value.0)
        .is_ok()
}

/// The classing result for one function. Class 0 is the escaped world;
/// each private alloca owns the class at its position in the sorted
/// list plus one.
#[derive(Debug)]
pub struct ClassAssignment {
    /// Sorted by id.
    private_allocas: Vec<ValueId>,
}

impl ClassAssignment {
    pub fn class_of_origin(&self, origin: MemoryOrigin) -> AliasClassId {
        match origin {
            MemoryOrigin::Alloca(node) => {
                match self
                    .private_allocas
                    .binary_search_by_key(&node.0, |value| value.0)
                {
                    Ok(index) => AliasClassId(index as u32 + 1),
                    Err(_) => AliasClassId(0),
                }
            }
            // Params, globals, call results, and loaded pointers are
            // all the caller's world at this tier.
            MemoryOrigin::Param(_)
            | MemoryOrigin::Global(_)
            | MemoryOrigin::CallResult(_)
            | MemoryOrigin::Func(_)
            | MemoryOrigin::Unknown => AliasClassId(0),
            // Non-pointers have no class to ask for, and resolution
            // leaves no Derived behind.
            MemoryOrigin::None => {
                debug_assert!(false, "class requested for a non-pointer origin");
                AliasClassId(0)
            }
            MemoryOrigin::Derived(_) => {
                unreachable!("classing runs after resolve_origins")
            }
        }
    }

    /// Total classes including class 0.
    pub fn class_count(&self) -> u32 {
        self.private_allocas.len() as u32 + 1
    }

    pub fn private_allocas(&self) -> &[ValueId] {
        &self.private_allocas
    }
}

impl FunctionFacts {
    /// Decide the function's alias classes: every alloca that neither
    /// escapes (per the inputs tier) nor is folded by its own uses
    /// keeps a private class; everything else is class 0.
    pub(crate) fn compute_classes(&self, inputs: &impl ClassingInputs) -> ClassAssignment {
        let private_allocas: Vec<ValueId> = self
            .allocas
            .iter()
            .copied()
            .filter(|alloca| {
                !inputs.origin_escaped(*alloca)
                    && !contains_id(&self.folded_volatile_atomic, *alloca)
                    && !contains_id(&self.folded_multi_address, *alloca)
            })
            .collect();
        ClassAssignment { private_allocas }
    }

    /// Where the function's allocas went and why, for the module
    /// census. Buckets are disjoint, assigned in priority order, so
    /// they sum to total_allocas.
    pub(crate) fn classing_census(&self) -> ClassingCensus {
        let mut census = ClassingCensus::default();
        census.total_allocas = self.allocas.len() as u64;
        for &alloca in &self.allocas {
            if contains_id(&self.escaped_origins, alloca) {
                census.escaped += 1;
            } else if contains_id(&self.call_retained, alloca) {
                census.retained_by_call += 1;
            } else if contains_id(&self.folded_volatile_atomic, alloca) {
                census.folded_volatile_atomic += 1;
            } else if contains_id(&self.folded_multi_address, alloca) {
                census.folded_multi_address += 1;
            } else {
                census.private += 1;
            }
        }
        census
    }
}

/// Per-cause alloca buckets, disjoint by priority (an alloca that both
/// escapes and is passed to a call counts once, as escaped).
#[derive(Debug, Default, Clone, Copy)]
pub struct ClassingCensus {
    pub total_allocas: u64,
    /// Hard escapes: stored, returned, int-cast, or widened away.
    pub escaped: u64,
    /// Only handed to calls -- the headroom the summaries tier
    /// (captured_params) will reclaim.
    pub retained_by_call: u64,
    pub folded_volatile_atomic: u64,
    pub folded_multi_address: u64,
    /// Survivors: one private class each.
    pub private: u64,
}

impl ClassingCensus {
    pub fn accumulate(&mut self, other: ClassingCensus) {
        self.total_allocas += other.total_allocas;
        self.escaped += other.escaped;
        self.retained_by_call += other.retained_by_call;
        self.folded_volatile_atomic += other.folded_volatile_atomic;
        self.folded_multi_address += other.folded_multi_address;
        self.private += other.private;
    }
}

#[cfg(test)]
mod test {
    use super::{ClassingInputs, FactsClassingInputs};
    use crate::rvsdg::{
        IntrinsicOp, Linkage, MemoryOrdering, RVSDGMod, RegionId, ValueId,
        memory_alias::origin::MemoryOrigin,
        types::{I32, PtrType, TypeRef},
    };

    fn test_module() -> (RVSDGMod, TypeRef) {
        let mut module = RVSDGMod::new_host(String::from("test"));
        let i32_ptr = TypeRef::Ptr(module.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        }));
        (module, i32_ptr)
    }

    // int f() {
    //     int a[4];            // stored -> hard escape
    //     int b[4];            // passed to g -> call retains it
    //     int c[4];            // only direct use -> private
    // }
    #[test]
    fn facts_inputs_apply_every_call_retains() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(String::from("g"), &[i32_ptr], &[I32], Linkage::Internal);
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        let mut alloca_nodes = None;
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let escaped_by_store = rb.alloca(I32, four, i32_ptr, None);
                let escaped_by_call = rb.alloca(I32, four, i32_ptr, None);
                let private = rb.alloca(I32, four, i32_ptr, None);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                rb.store(slot, escaped_by_store, None, false);
                rb.call(callee, &[escaped_by_call]);
                let value = rb.load(private, I32, None, false);
                rb.store(private, value, None, false);
                alloca_nodes = Some((escaped_by_store, escaped_by_call, private, slot));
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let (escaped_by_store, escaped_by_call, private, slot) = alloca_nodes.unwrap();

        let graph = module.graphs[function.0 as usize].as_ref().unwrap();
        let inputs = FactsClassingInputs::new(&module.facts[function.0 as usize]);

        let node_of = |projection| match graph.get_memory_origin(projection) {
            MemoryOrigin::Alloca(node) => node,
            other => panic!("expected an alloca origin, got {other:?}"),
        };
        assert!(inputs.origin_escaped(node_of(escaped_by_store)));
        assert!(
            inputs.origin_escaped(node_of(escaped_by_call)),
            "a pointer handed to a call is retained at this tier"
        );
        assert!(!inputs.origin_escaped(node_of(private)));
        assert!(
            !inputs.origin_escaped(node_of(slot)),
            "holding an escaping pointer does not escape the holder"
        );
    }

    /// The alloca pointer projections a test body smuggles out for
    /// assertions.
    struct FoldTestPtrs {
        plain: ValueId,
        hot: ValueId,
        shared: ValueId,
    }

    // int f() {
    //     int plain[4];  x = plain[0];                    // untouched
    //     int hot[4];    volatile_store(hot, 1);          // folded
    //     int shared[4]; atomic_store(shared, 1);         // folded
    // }
    #[test]
    fn volatile_and_atomic_accesses_fold_their_slot() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        let mut smuggled = None;
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let plain = rb.alloca(I32, four, i32_ptr, None);
                let hot = rb.alloca(I32, four, i32_ptr, None);
                let shared = rb.alloca(I32, four, i32_ptr, None);
                let value = rb.load(plain, I32, None, false);
                rb.store(hot, value, None, true); // volatile
                rb.atomic_store(shared, value, MemoryOrdering::Relaxed, None, false);
                smuggled = Some(FoldTestPtrs { plain, hot, shared });
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let ptrs = smuggled.unwrap();
        let (plain, hot, shared) = (ptrs.plain, ptrs.hot, ptrs.shared);

        let graph = module.graphs[function.0 as usize].as_ref().unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(facts.allocas.len(), 3);

        let node_of = |projection| match graph.get_memory_origin(projection) {
            MemoryOrigin::Alloca(node) => node,
            other => panic!("expected an alloca origin, got {other:?}"),
        };
        assert!(!facts.folded_volatile_atomic.contains(&node_of(plain)));
        assert!(facts.folded_volatile_atomic.contains(&node_of(hot)));
        assert!(facts.folded_volatile_atomic.contains(&node_of(shared)));
        assert!(facts.folded_multi_address.is_empty());
    }

    /// The alloca pointer projections the intrinsic test smuggles out.
    struct IntrinsicTestPtrs {
        set_only: ValueId,
        copy_dest: ValueId,
        copy_source: ValueId,
        from_param: ValueId,
    }

    // memset(a) keeps a's class (one address); memcpy(b, c) folds both;
    // memcpy(d, param) folds d (a class-0 source makes it multi-chain).
    #[test]
    fn multi_address_intrinsics_fold_participants() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[i32_ptr], &[I32], Linkage::Internal);
        let mut smuggled = None;
        module
            .define_fn(function, |rb| {
                let param_ptr = rb.param(0);
                let four = rb.const_i32(4);
                let set_only = rb.alloca(I32, four, i32_ptr, None);
                let copy_dest = rb.alloca(I32, four, i32_ptr, None);
                let copy_source = rb.alloca(I32, four, i32_ptr, None);
                let from_param = rb.alloca(I32, four, i32_ptr, None);
                let byte_count = rb.const_i64(16);
                let fill = rb.const_i32(0);
                rb.intrinsic_void(IntrinsicOp::MemSet, &[set_only, fill, byte_count]);
                rb.intrinsic_void(IntrinsicOp::MemCopy, &[copy_dest, copy_source, byte_count]);
                rb.intrinsic_void(IntrinsicOp::MemCopy, &[from_param, param_ptr, byte_count]);
                smuggled = Some(IntrinsicTestPtrs {
                    set_only,
                    copy_dest,
                    copy_source,
                    from_param,
                });
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let ptrs = smuggled.unwrap();
        let (set_only, copy_dest, copy_source, from_param) = (
            ptrs.set_only,
            ptrs.copy_dest,
            ptrs.copy_source,
            ptrs.from_param,
        );

        let graph = module.graphs[function.0 as usize].as_ref().unwrap();
        let facts = &module.facts[function.0 as usize];

        let node_of = |projection| match graph.get_memory_origin(projection) {
            MemoryOrigin::Alloca(node) => node,
            other => panic!("expected an alloca origin, got {other:?}"),
        };
        assert!(
            !facts.folded_multi_address.contains(&node_of(set_only)),
            "one address, one chain: memset stays in its slot's class"
        );
        assert!(facts.folded_multi_address.contains(&node_of(copy_dest)));
        assert!(facts.folded_multi_address.contains(&node_of(copy_source)));
        assert!(facts.folded_multi_address.contains(&node_of(from_param)));
        assert!(facts.folded_volatile_atomic.is_empty());
    }

    /// The alloca pointer projections the end-to-end test smuggles out.
    struct AssignmentTestPtrs {
        stored: ValueId,
        passed: ValueId,
        hot: ValueId,
        first_private: ValueId,
        second_private: ValueId,
    }

    // The full 3.1 path: escaped-by-store, escaped-by-call, and
    // volatile-folded allocas land in class 0; the two clean allocas
    // get private classes numbered by node order; every non-alloca
    // origin is class 0.
    #[test]
    fn compute_classes_assigns_private_classes() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(String::from("g"), &[i32_ptr], &[I32], Linkage::Internal);
        let function = module.declare_fn(String::from("f"), &[i32_ptr], &[I32], Linkage::Internal);
        let mut smuggled = None;
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let stored = rb.alloca(I32, four, i32_ptr, None);
                let first_private = rb.alloca(I32, four, i32_ptr, None);
                let passed = rb.alloca(I32, four, i32_ptr, None);
                let second_private = rb.alloca(I32, four, i32_ptr, None);
                let hot = rb.alloca(i32_ptr, four, i32_ptr, None);
                rb.store(hot, stored, None, true); // volatile store of &stored
                rb.call(callee, &[passed]);
                let value = rb.load(first_private, I32, None, false);
                rb.store(second_private, value, None, false);
                smuggled = Some(AssignmentTestPtrs {
                    stored,
                    passed,
                    hot,
                    first_private,
                    second_private,
                });
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let ptrs = smuggled.unwrap();

        let graph = module.graphs[function.0 as usize].as_ref().unwrap();
        let facts = &module.facts[function.0 as usize];
        let inputs = FactsClassingInputs::new(facts);
        let assignment = facts.compute_classes(&inputs);

        assert_eq!(assignment.class_count(), 3, "class 0 plus two privates");
        let class_of = |projection| assignment.class_of_origin(graph.get_memory_origin(projection));
        assert_eq!(class_of(ptrs.stored).0, 0, "stored as data: escaped");
        assert_eq!(class_of(ptrs.passed).0, 0, "handed to a call: retained");
        assert_eq!(class_of(ptrs.hot).0, 0, "volatile-accessed slot folds");
        // Private classes number by node order: first_private was
        // created before second_private.
        assert_eq!(class_of(ptrs.first_private).0, 1);
        assert_eq!(class_of(ptrs.second_private).0, 2);
        // Non-alloca origins are the caller's world.
        let parameter = graph.region_params(RegionId(0))[0];
        assert_eq!(
            assignment
                .class_of_origin(graph.get_memory_origin(parameter))
                .0,
            0,
            "a parameter's pointee is class 0 at this tier"
        );

        // The census buckets the same function: one escaped (stored),
        // one retained by call, one volatile-folded, two private --
        // disjoint and summing to the five allocas.
        let census = facts.classing_census();
        assert_eq!(
            census.private,
            u64::from(assignment.class_count() - 1),
            "census survivors and classing privates must not drift"
        );
        assert_eq!(census.total_allocas, 5);
        assert_eq!(census.escaped, 1);
        assert_eq!(census.retained_by_call, 1);
        assert_eq!(census.folded_volatile_atomic, 1);
        assert_eq!(census.folded_multi_address, 0);
        assert_eq!(census.private, 2);
    }
}
