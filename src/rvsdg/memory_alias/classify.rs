//! Event classification: after origin resolution, classify the
//! recorded events through the FINAL tags into `FunctionFacts`. Each pass
//! handles one event list, so the pieces review and test alone;
//! `FunctionFacts::finalize` sorts and dedups the accumulated lists
//! once, after the last pass has appended.

use smallvec::SmallVec;

use crate::rvsdg::{
    ValueId,
    func::MemReadWrite,
    function_graph::FunctionGraph,
    memory_alias::{CallFact, FunctionFacts, LocalReturn, origin::MemoryOrigin},
    module_tables::ModuleTables,
    types::TypeRef,
    value::ValueKind,
};

/// Where an escaping value was seen. The one behavioural difference:
/// a RETURNED CallResult records only `local_return`, never an
/// anonymous deferred escape -- the barrier's Fresh rule is "the
/// result's only escaping position was the return itself", which an
/// indistinguishable escape entry would destroy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EscapeContext {
    Hard,
    Return,
}

impl FunctionGraph {
    /// Classify the hard-escape events. See
    /// [`FunctionGraph::classify_escaping_value`] for the per-value
    /// rules.
    pub(crate) fn classify_escape_events(&self, tables: &ModuleTables, facts: &mut FunctionFacts) {
        debug_assert!(
            self.mem_facts
                .call_sites
                .is_sorted_by_key(|call_site| call_site.0),
            "call sites must be in construction order for index lookup"
        );
        for &value in &self.mem_facts.escape_events {
            self.classify_escaping_value(tables, value, EscapeContext::Hard, facts);
        }
    }

    /// One escaping value, classified by its resolved origin:
    ///
    ///   Alloca(node)       -> escaped_origins
    ///   Param(index)       -> local_captured (the capture seed)
    ///   Global(global)     -> address_taken_global (untracked forever)
    ///   Func(function)     -> address_taken_functions (externally
    ///                         callable; the Func origin survives
    ///                         derivation, so `cond ? &f : &g` widening
    ///                         names both)
    ///   CallResult(call)   -> deferred_escapes as a call index (the
    ///                         barrier judges it), EXCEPT for returns
    ///                         (see EscapeContext)
    ///   Unknown            -> nothing named to record
    ///
    /// The one kind needing more than its origin: a NON-pointer
    /// ConstPoolRef (an aggregate constant like `{ &f, &g_var }`) has
    /// no single base -- its payloads are walked recursively, since
    /// packed constant addresses have no insert site to catch them.
    ///
    /// Deferred escapes are stored as indices into the call-site list,
    /// which is ascending by construction order -- the same order the
    /// CallFact list is built in, so the index survives the events.
    fn classify_escaping_value(
        &self,
        tables: &ModuleTables,
        value: ValueId,
        context: EscapeContext,
        facts: &mut FunctionFacts,
    ) {
        if let ValueKind::ConstPoolRef(constant) = self.get_value_kind(value) {
            if !matches!(self.get_value_type(value), TypeRef::Ptr(_)) {
                collect_constant_addresses(tables, *constant, facts);
                return;
            }
        }
        match self.get_memory_origin(value) {
            MemoryOrigin::Alloca(node) => facts.escaped_origins.push(node),
            MemoryOrigin::Param(index) => facts.local_captured.set_retained(index),
            MemoryOrigin::Global(global) => facts.address_taken_global.push(global),
            MemoryOrigin::Func(function) => facts.address_taken_functions.push(function),
            MemoryOrigin::CallResult(call) => {
                if context == EscapeContext::Hard {
                    let call_index = call_index(&self.mem_facts.call_sites, call);
                    facts.deferred_escapes.push(call_index);
                }
            }
            MemoryOrigin::Unknown => {}
            // The escape recorder gates on pointer types (and the
            // ConstPoolRef case returned above); resolution leaves no
            // Derived behind.
            MemoryOrigin::None => {
                debug_assert!(false, "non-pointer value in escape events")
            }
            MemoryOrigin::Derived(_) => {
                unreachable!("classification runs after resolve_origins")
            }
        }
    }

    /// Classify the access events: each memory op contributes its
    /// effect through its ADDRESS operand's origin.
    ///
    ///   Alloca      -> nothing: a caller cannot name the callee's
    ///                  frame. (The op still gets the alloca's CLASS at
    ///                  split time -- facts record only what callers
    ///                  can observe.)
    ///   Param(k)    -> local_param_effects entry k
    ///   Global(g)   -> the (g, effect) pair list
    ///   CallResult  -> deferred_accesses; the barrier resolves it
    ///                  through the callee's return provenance
    ///   Unknown     -> local_other
    ///
    /// Memory-class intrinsics have no single address (memcpy touches a
    /// source AND a destination), so every pointer argument contributes
    /// conservatively as ReadAndWrite. Fences carry no address; their
    /// ordering is a split-time concern, invisible to callers.
    ///
    /// This pass also records the classing fold sets, since it is
    /// already standing at every memory op with resolved origins:
    ///
    ///  - A VOLATILE or ATOMIC access folds its alloca's slot for the
    ///    whole function: one op has one chain membership, so pinning
    ///    only the offending site would leave two accesses to the same
    ///    slot on different chains with no edge between them.
    ///  - A multi-address intrinsic whose pointer arguments span more
    ///    than one class (each distinct alloca is one; any non-alloca
    ///    pointer collectively counts as class 0) folds every alloca it
    ///    touches: the op carries a single state operand and cannot be
    ///    a member of two chains.
    ///
    /// Fences pin to class 0 and would fold global classes, which do
    /// not exist at this tier.
    pub(crate) fn classify_access_events(&self, facts: &mut FunctionFacts) {
        for &op in &self.mem_facts.access_events {
            let kind = self.get_value_kind(op);
            if let Some(access) = kind.memory_access() {
                let folds_slot = match kind {
                    ValueKind::Load { volatile, .. } | ValueKind::Store { volatile, .. } => {
                        *volatile
                    }
                    ValueKind::AtomicLoad { .. }
                    | ValueKind::AtomicStore { .. }
                    | ValueKind::AtomicReadModifyWrite { .. }
                    | ValueKind::CompareAndSwap { .. } => true,
                    _ => false,
                };
                if folds_slot {
                    if let MemoryOrigin::Alloca(node) = self.get_memory_origin(access.addr) {
                        facts.folded_volatile_atomic.push(node);
                    }
                }
                self.classify_access(access.addr, access.effect, facts);
            } else if let ValueKind::Intrinsic { args, .. } = kind {
                let args = *args;
                let mut alloca_targets: SmallVec<[ValueId; 2]> = SmallVec::new();
                let mut touches_class_zero = false;
                for arg_index in 0..args.len {
                    let arg = self.value_pool.get(args)[arg_index as usize];
                    if !matches!(self.get_value_type(arg), TypeRef::Ptr(_)) {
                        continue;
                    }
                    match self.get_memory_origin(arg) {
                        MemoryOrigin::Alloca(node) => {
                            if !alloca_targets.contains(&node) {
                                alloca_targets.push(node);
                            }
                        }
                        _ => touches_class_zero = true,
                    }
                    self.classify_access(arg, MemReadWrite::ReadAndWrite, facts);
                }
                if alloca_targets.len() + usize::from(touches_class_zero) > 1 {
                    facts
                        .folded_multi_address
                        .extend_from_slice(&alloca_targets);
                }
            } else {
                debug_assert!(
                    matches!(kind, ValueKind::Fence { .. }),
                    "unhandled op kind in access events"
                );
            }
        }
    }

    fn classify_access(&self, addr: ValueId, effect: MemReadWrite, facts: &mut FunctionFacts) {
        match self.get_memory_origin(addr) {
            MemoryOrigin::Alloca(_) => {}
            MemoryOrigin::Param(index) => facts.local_param_effects.join(index, effect),
            MemoryOrigin::Global(global) => {
                join_effect_pair(&mut facts.global_access, global, effect);
            }
            MemoryOrigin::CallResult(call) => {
                let call_index = call_index(&self.mem_facts.call_sites, call);
                join_effect_pair(&mut facts.deferred_accesses, call_index, effect);
            }
            MemoryOrigin::Unknown => facts.local_other = facts.local_other.join(effect),
            // Loading or storing THROUGH a function's address is
            // malformed input; price it as unknown memory in release.
            MemoryOrigin::Func(_) => {
                debug_assert!(false, "memory access through a function address");
                facts.local_other = facts.local_other.join(effect);
            }
            MemoryOrigin::None => {
                debug_assert!(false, "memory access through a non-pointer address")
            }
            MemoryOrigin::Derived(_) => {
                unreachable!("classification runs after resolve_origins")
            }
        }
    }

    /// Build the id-free CallFacts and classify the returned values.
    ///
    /// Known callee: `arg_provenance` is indexed by the callee's
    /// DECLARED parameter positions -- a pointer parameter gets the
    /// argument's origin, a non-pointer (or unfilled) position gets
    /// None. Pointer arguments beyond the declared list (varargs) or at
    /// a position whose DECLARED parameter is not pointer-typed are not
    /// translated: they classify as hard escapes instead, because the
    /// callee's own facts gated on its declared types and never saw a
    /// pointer there. Indirect callee: positional over the actual
    /// arguments; the barrier applies the external node's TOP, so every
    /// named origin in the list escapes anyway.
    ///
    /// Alloca-origin arguments also land in `call_retained` -- the
    /// facts tier's "every call retains" set -- recorded here where the
    /// origins are already in hand.
    ///
    /// Returns classify like hard escapes except the CallResult arm
    /// (see EscapeContext); `local_return` records what the single
    /// pointer-typed result traces to, when there is exactly one.
    pub(crate) fn classify_calls_and_returns(
        &self,
        tables: &ModuleTables,
        facts: &mut FunctionFacts,
    ) {
        for &call in &self.mem_facts.call_sites {
            match self.get_value_kind(call) {
                ValueKind::Call { fn_id, args, .. } => {
                    let callee = *fn_id;
                    let args = *args;
                    let declared_params = &tables.get_function(callee).params;
                    let args_slice = self.value_pool.get(args);
                    let mut arg_provenance = SmallVec::new();
                    for (position, declared) in declared_params.iter().enumerate() {
                        let Some(&argument) = args_slice.get(position) else {
                            arg_provenance.push(MemoryOrigin::None);
                            continue;
                        };
                        if !matches!(self.get_value_type(argument), TypeRef::Ptr(_)) {
                            arg_provenance.push(MemoryOrigin::None);
                            continue;
                        }
                        if !matches!(declared.ty, TypeRef::Ptr(_)) {
                            // A pointer passed where the callee DECLARED
                            // a non-pointer: the callee's recorder gated
                            // on its declared type and never saw a
                            // pointer, so its facts cannot answer for
                            // this argument. Hard-escape, like a vararg.
                            self.classify_escaping_value(
                                tables,
                                argument,
                                EscapeContext::Hard,
                                facts,
                            );
                            arg_provenance.push(MemoryOrigin::None);
                            continue;
                        }
                        let origin = self.get_memory_origin(argument);
                        if let MemoryOrigin::Alloca(node) = origin {
                            facts.call_retained.push(node);
                        }
                        arg_provenance.push(origin);
                    }
                    for extra_index in declared_params.len()..args_slice.len() {
                        let extra = args_slice[extra_index];
                        if matches!(self.get_value_type(extra), TypeRef::Ptr(_)) {
                            self.classify_escaping_value(tables, extra, EscapeContext::Hard, facts);
                        }
                    }
                    facts.calls.push(CallFact {
                        callee: Some(callee),
                        arg_provenance,
                    });
                }
                ValueKind::CallIndirect { args, .. } => {
                    let args = *args;
                    let args_slice = self.value_pool.get(args);
                    let mut arg_provenance = SmallVec::new();
                    for &argument in args_slice {
                        let origin = if matches!(self.get_value_type(argument), TypeRef::Ptr(_)) {
                            self.get_memory_origin(argument)
                        } else {
                            MemoryOrigin::None
                        };
                        if let MemoryOrigin::Alloca(node) = origin {
                            facts.call_retained.push(node);
                        }
                        arg_provenance.push(origin);
                    }
                    facts.calls.push(CallFact {
                        callee: None,
                        arg_provenance,
                    });
                }
                _ => debug_assert!(false, "non-call value in call sites"),
            }
        }

        let mut pointer_return: Option<ValueId> = None;
        let mut multiple_pointer_returns = false;
        for &value in &self.mem_facts.returns {
            if !matches!(self.get_value_type(value), TypeRef::Ptr(_)) {
                continue;
            }
            self.classify_escaping_value(tables, value, EscapeContext::Return, facts);
            if pointer_return.replace(value).is_some() {
                multiple_pointer_returns = true;
            }
        }
        facts.local_return = match pointer_return {
            Some(value) if !multiple_pointer_returns => match self.get_memory_origin(value) {
                MemoryOrigin::Param(index) if index <= u8::MAX as u32 => {
                    LocalReturn::Param(index as u8)
                }
                MemoryOrigin::CallResult(call) => {
                    LocalReturn::CallResult(call_index(&self.mem_facts.call_sites, call))
                }
                _ => LocalReturn::Unknown,
            },
            _ => LocalReturn::Unknown,
        };
    }
}

/// Recursively collect the global and function addresses inside a pool
/// constant: aggregate constants (`{ &f, &g_var }`) smuggle addresses
/// with no insert site, so an escaping one address-takes everything it
/// holds.
fn collect_constant_addresses(
    tables: &ModuleTables,
    constant: crate::rvsdg::constant::ConstId,
    facts: &mut FunctionFacts,
) {
    use crate::rvsdg::constant::ConstantKind;
    match &tables.constants.get(constant).kind {
        ConstantKind::GlobalAddr(global) => facts.address_taken_global.push(*global),
        ConstantKind::FuncAddr(function) => facts.address_taken_functions.push(*function),
        ConstantKind::GetElementPointer { base, .. } => {
            collect_constant_addresses(tables, *base, facts)
        }
        ConstantKind::Cast { operand, .. } => collect_constant_addresses(tables, *operand, facts),
        ConstantKind::Aggregate(elements) => {
            for element_index in 0..elements.len {
                let element =
                    tables.constants.get_aggregate_elements(*elements)[element_index as usize];
                collect_constant_addresses(tables, element, facts);
            }
        }
        _ => {}
    }
}

/// The position of `call` in the construction-ordered call-site list:
/// the index CallFacts are built in, so it survives the value-id world.
fn call_index(call_sites: &[ValueId], call: ValueId) -> u32 {
    call_sites
        .binary_search_by_key(&call.0, |call_site| call_site.0)
        .expect("CallResult origin names a recorded call site") as u32
}

impl FunctionFacts {
    /// Sort and dedup the accumulated lists, once, after the last
    /// classification pass has appended (returns append to the same
    /// lists the escape pass does, so sorting inside either would be
    /// premature).
    pub(crate) fn finalize(&mut self) {
        debug_assert!(
            self.allocas.is_sorted_by_key(|value| value.0),
            "allocas record in construction order"
        );
        sort_dedup_by_id(&mut self.escaped_origins, |value: &ValueId| value.0);
        sort_dedup_by_id(&mut self.folded_volatile_atomic, |value: &ValueId| value.0);
        sort_dedup_by_id(&mut self.folded_multi_address, |value: &ValueId| value.0);
        sort_dedup_by_id(&mut self.call_retained, |value: &ValueId| value.0);
        sort_dedup_by_id(&mut self.address_taken_global, |global| global.0);
        sort_dedup_by_id(&mut self.address_taken_functions, |function| function.0);
        sort_dedup_by_id(&mut self.deferred_escapes, |index| *index);
        self.global_access
            .sort_unstable_by_key(|(global, _)| global.0);
        self.deferred_accesses
            .sort_unstable_by_key(|(call_index, _)| *call_index);
    }
}

/// Sort by a numeric key and drop duplicates: escape events routinely
/// record one origin several times (a pointer stored twice), and the
/// consumers want membership, not multiplicity.
fn sort_dedup_by_id<Element, Key: Ord + Copy>(
    list: &mut SmallVec<impl smallvec::Array<Item = Element>>,
    key: impl Fn(&Element) -> Key,
) {
    list.sort_unstable_by_key(&key);
    list.dedup_by_key(|element| key(element));
}

/// Join `effect` into the pair keyed `key`, or start a new pair: two
/// accesses through one global or one call result must accumulate to
/// their lattice join, never overwrite.
fn join_effect_pair<Key: PartialEq + Copy>(
    pairs: &mut SmallVec<impl smallvec::Array<Item = (Key, MemReadWrite)>>,
    key: Key,
    effect: MemReadWrite,
) {
    if let Some((_, existing)) = pairs.iter_mut().find(|(pair_key, _)| *pair_key == key) {
        *existing = existing.join(effect);
    } else {
        pairs.push((key, effect));
    }
}

#[cfg(test)]
mod test {
    use crate::rvsdg::{
        ICmpPred, Linkage, RVSDGMod,
        memory_alias::{FunctionFacts, LocalReturn, origin::MemoryOrigin},
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

    // void f(int *param_ptr) {
    //     int a[4];
    //     slot = &a;  slot = &a;      // alloca escapes (twice: dedup)
    //     slot = param_ptr;           // param 0 retained
    //     slot = (void*)f;            // f's address taken
    // }
    #[test]
    fn escape_events_classify_by_kind_then_origin() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[i32_ptr], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        module
            .define_fn(function, |rb| {
                let param_ptr = rb.param(0);
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                rb.store(slot, alloca_a, None, false);
                rb.store(slot, alloca_a, None, false); // duplicate escape
                rb.store(slot, param_ptr, None, false);
                let own_address = rb.func_addr(function, i32_ptr);
                rb.store(slot, own_address, None, false);

                rb.graph.resolve_origins();
                let tables_view = &*rb.module_tables;
                rb.graph
                    .classify_escape_events(tables_view, &mut classified);
                classified.finalize();

                // One escaped origin despite two stores; it is the
                // alloca NODE the pointer projection traces to.
                assert_eq!(classified.escaped_origins.len(), 1);
                assert_eq!(
                    rb.graph.get_memory_origin(alloca_a),
                    MemoryOrigin::Alloca(classified.escaped_origins[0])
                );
                assert!(classified.local_captured.is_retained(0));
                assert!(!classified.local_captured.is_retained(1));
                assert_eq!(classified.address_taken_functions.as_slice(), &[function]);
                assert!(classified.address_taken_global.is_empty());
                assert!(classified.deferred_escapes.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // Storing a LOADED pointer names nothing: Unknown classifies to no
    // fact at all -- it is already class 0.
    #[test]
    fn unknown_escape_records_nothing() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                let loaded_ptr = rb.load(slot, i32_ptr, None, false);
                rb.store(slot, loaded_ptr, None, false);

                rb.graph.resolve_origins();
                let tables_view = &*rb.module_tables;
                rb.graph
                    .classify_escape_events(tables_view, &mut classified);
                classified.finalize();

                assert!(classified.escaped_origins.is_empty());
                assert_eq!(
                    classified.local_captured,
                    super::super::CapturedParams::EMPTY
                );
                assert!(classified.address_taken_functions.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // int f(int *param_ptr) {
    //     int a[4];
    //     x = param_ptr[0]; param_ptr[0] = x;   // param 0: RW
    //     y = a[0]; a[0] = y;                   // frame-private: nothing
    //     z = *loaded;                          // Unknown addr: other Read
    //     g = 1; use(g);                        // global: RW pair
    // }
    #[test]
    fn accesses_classify_by_address_origin() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[i32_ptr], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        let global = crate::rvsdg::GlobalId(7); // classification reads
        // origins only; no table entry is needed for this unit test.
        module
            .define_fn(function, |rb| {
                let param_ptr = rb.param(0);
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);

                let from_param = rb.load(param_ptr, I32, None, false);
                rb.store(param_ptr, from_param, None, false);
                let from_alloca = rb.load(alloca_a, I32, None, false);
                rb.store(alloca_a, from_alloca, None, false);
                let loaded_ptr = rb.load(slot, i32_ptr, None, false);
                let _through_unknown = rb.load(loaded_ptr, I32, None, false);
                let global_ptr = rb.global_ref(global, i32_ptr);
                rb.store(global_ptr, from_param, None, false);
                let _from_global = rb.load(global_ptr, I32, None, false);
                rb.fence(crate::rvsdg::MemoryOrdering::AcquireRelease);

                rb.graph.resolve_origins();
                rb.graph.classify_access_events(&mut classified);
                classified.finalize();

                use crate::rvsdg::func::MemReadWrite;
                assert_eq!(
                    classified.local_param_effects.get(0),
                    MemReadWrite::ReadAndWrite
                );
                assert_eq!(classified.local_param_effects.get(1), MemReadWrite::None);
                // The unknown-address load is the ONLY thing in other:
                // frame-private traffic (a, slot) contributes nothing.
                assert_eq!(classified.local_other, MemReadWrite::ReadOnly);
                assert_eq!(
                    classified.global_access.as_slice(),
                    &[(global, MemReadWrite::ReadAndWrite)]
                );
                assert!(classified.deferred_accesses.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // Accesses through a call's returned pointer defer to the barrier,
    // keyed by call index, with effects joined.
    #[test]
    fn call_result_accesses_defer_with_joined_effect() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(
            String::from("returns_ptr"),
            &[],
            &[i32_ptr],
            Linkage::Internal,
        );
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        module
            .define_fn(function, |rb| {
                let call_results = rb.call(callee, &[]);
                let returned_ptr = call_results.result(0);
                let loaded = rb.load(returned_ptr, I32, None, false);
                rb.store(returned_ptr, loaded, None, false);

                rb.graph.resolve_origins();
                rb.graph.classify_access_events(&mut classified);
                classified.finalize();

                use crate::rvsdg::func::MemReadWrite;
                assert_eq!(
                    classified.deferred_accesses.as_slice(),
                    &[(0u32, MemReadWrite::ReadAndWrite)]
                );
                assert_eq!(classified.local_other, MemReadWrite::None);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // memcpy-class intrinsics classify every pointer argument
    // conservatively as ReadAndWrite; non-pointer arguments and
    // frame-private pointers contribute nothing.
    #[test]
    fn intrinsic_pointer_args_classify_conservatively() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[i32_ptr], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        module
            .define_fn(function, |rb| {
                let dest_param = rb.param(0);
                let four = rb.const_i32(4);
                let source_alloca = rb.alloca(I32, four, i32_ptr, None);
                let byte_count = rb.const_i64(16);
                rb.intrinsic_void(
                    crate::rvsdg::IntrinsicOp::MemCopy,
                    &[dest_param, source_alloca, byte_count],
                );

                rb.graph.resolve_origins();
                rb.graph.classify_access_events(&mut classified);
                classified.finalize();

                use crate::rvsdg::func::MemReadWrite;
                assert_eq!(
                    classified.local_param_effects.get(0),
                    MemReadWrite::ReadAndWrite
                );
                assert_eq!(classified.local_other, MemReadWrite::None);
                assert!(classified.global_access.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // int g(int x, int *p, int y);  f() { g(1, &a, 2); }
    // arg_provenance is indexed by DECLARED positions: None for the
    // non-pointer slots, the argument's origin for the pointer slot.
    #[test]
    fn call_fact_indexes_declared_positions() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(
            String::from("g"),
            &[I32, i32_ptr, I32],
            &[I32],
            Linkage::Internal,
        );
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let one = rb.const_i32(1);
                let two = rb.const_i32(2);
                rb.call(callee, &[one, alloca_a, two]);

                rb.graph.resolve_origins();
                let tables_view = &*rb.module_tables;
                rb.graph
                    .classify_calls_and_returns(tables_view, &mut classified);
                classified.finalize();

                assert_eq!(classified.calls.len(), 1);
                let call_fact = &classified.calls[0];
                assert_eq!(call_fact.callee, Some(callee));
                assert_eq!(call_fact.arg_provenance.len(), 3);
                assert_eq!(call_fact.arg_provenance[0], MemoryOrigin::None);
                assert_eq!(
                    call_fact.arg_provenance[1],
                    rb.graph.get_memory_origin(alloca_a)
                );
                assert_eq!(call_fact.arg_provenance[2], MemoryOrigin::None);
                assert!(classified.escaped_origins.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // A pointer argument BEYOND the declared list (varargs shape) is
    // not translated: it hard-escapes instead.
    #[test]
    fn vararg_pointer_hard_escapes() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(String::from("g"), &[I32], &[I32], Linkage::Internal);
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        let mut classified = FunctionFacts::empty();
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let one = rb.const_i32(1);
                rb.call(callee, &[one, alloca_a]);

                rb.graph.resolve_origins();
                let tables_view = &*rb.module_tables;
                rb.graph
                    .classify_calls_and_returns(tables_view, &mut classified);
                classified.finalize();

                let call_fact = &classified.calls[0];
                assert_eq!(call_fact.arg_provenance.len(), 1, "declared width only");
                assert_eq!(call_fact.arg_provenance[0], MemoryOrigin::None);
                assert_eq!(classified.escaped_origins.len(), 1);
                assert_eq!(
                    rb.graph.get_memory_origin(alloca_a),
                    MemoryOrigin::Alloca(classified.escaped_origins[0])
                );
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // int g(int x);  f() { g(&a); }
    // A pointer passed at a position whose DECLARED parameter is not
    // pointer-typed: not translated (the callee's facts never saw a
    // pointer there), so it hard-escapes and is not call-retained.
    #[test]
    fn mismatched_pointer_position_hard_escapes() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(String::from("g"), &[I32], &[I32], Linkage::Internal);
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                rb.call(callee, &[alloca_a]);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(facts.calls.len(), 1);
        assert_eq!(
            facts.calls[0].arg_provenance.as_slice(),
            &[MemoryOrigin::None]
        );
        assert!(facts.call_retained.is_empty());
        assert_eq!(facts.escaped_origins.len(), 1, "the mismatched pointer");
    }

    // Returning a parameter: local_return = Param(0) AND the parameter
    // counts as retained. Returning a call result: local_return =
    // CallResult(0) and -- the Fresh-preserving property -- NO deferred
    // escape is recorded for it. Asserted through the production path:
    // define_fn records the returns and runs resolve_facts, the results
    // land in module.facts.
    #[test]
    fn returns_classify_with_fresh_preserving_call_result() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(
            String::from("returns_ptr"),
            &[],
            &[i32_ptr],
            Linkage::Internal,
        );
        let returns_param = module.declare_fn(
            String::from("returns_param"),
            &[i32_ptr],
            &[i32_ptr],
            Linkage::Internal,
        );
        module
            .define_fn(returns_param, |rb| Ok(vec![rb.param(0)]))
            .unwrap();
        let param_facts = &module.facts[returns_param.0 as usize];
        assert_eq!(param_facts.local_return, LocalReturn::Param(0));
        assert!(param_facts.local_captured.is_retained(0));

        let returns_fresh = module.declare_fn(
            String::from("returns_fresh"),
            &[],
            &[i32_ptr],
            Linkage::Internal,
        );
        module
            .define_fn(returns_fresh, |rb| {
                let call_results = rb.call(callee, &[]);
                Ok(vec![call_results.result(0)])
            })
            .unwrap();
        let fresh_facts = &module.facts[returns_fresh.0 as usize];
        assert_eq!(fresh_facts.local_return, LocalReturn::CallResult(0));
        assert!(
            fresh_facts.deferred_escapes.is_empty(),
            "a returned call result is not an anonymous escape"
        );
        assert_eq!(fresh_facts.calls.len(), 1);
        assert_eq!(fresh_facts.calls[0].callee, Some(callee));
    }

    // fp = cond ? &f : &g; *slot = fp;
    // The widening join discards two FUNCTION addresses: both must land
    // in address_taken_functions. This is the case a kind check on the
    // escaping value can never see -- only the Func origin surviving
    // derivation makes it work.
    #[test]
    fn func_addr_select_widens_into_address_taken() {
        let (mut module, i32_ptr) = test_module();
        let callee_a = module.declare_fn(String::from("fa"), &[], &[I32], Linkage::Internal);
        let callee_b = module.declare_fn(String::from("fb"), &[], &[I32], Linkage::Internal);
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                let address_a = rb.func_addr(callee_a, i32_ptr);
                let address_b = rb.func_addr(callee_b, i32_ptr);
                let zero = rb.const_i32(0);
                let flag = rb.icmp(ICmpPred::Eq, zero, zero);
                let selected = rb.ternary(flag, address_a, address_b, i32_ptr);
                rb.store(slot, selected, None, false);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(
            facts.address_taken_functions.as_slice(),
            &[callee_a, callee_b]
        );
    }

    // Widening must fire for EVERY named-origin kind it discards:
    // select(param, global) stored retains the param and address-takes
    // the global.
    #[test]
    fn widening_discards_param_and_global() {
        let (mut module, i32_ptr) = test_module();
        let global = crate::rvsdg::GlobalId(3);
        let function = module.declare_fn(String::from("f"), &[i32_ptr], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let param_ptr = rb.param(0);
                let four = rb.const_i32(4);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                let global_ptr = rb.global_ref(global, i32_ptr);
                let zero = rb.const_i32(0);
                let flag = rb.icmp(ICmpPred::Eq, zero, zero);
                let selected = rb.ternary(flag, param_ptr, global_ptr, i32_ptr);
                rb.store(slot, selected, None, false);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert!(facts.local_captured.is_retained(0));
        assert_eq!(facts.address_taken_global.as_slice(), &[global]);
    }

    // select(call result, alloca) stored: widening defers the call
    // result's escape to the barrier AND escapes the alloca.
    #[test]
    fn widening_discards_call_result_and_alloca() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(
            String::from("returns_ptr"),
            &[],
            &[i32_ptr],
            Linkage::Internal,
        );
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                let call_results = rb.call(callee, &[]);
                let zero = rb.const_i32(0);
                let flag = rb.icmp(ICmpPred::Eq, zero, zero);
                let selected = rb.ternary(flag, call_results.result(0), alloca_a, i32_ptr);
                rb.store(slot, selected, None, false);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(facts.deferred_escapes.as_slice(), &[0u32]);
        assert_eq!(facts.escaped_origins.len(), 1, "the discarded alloca");
    }

    // Storing an AGGREGATE pool constant `{ &f, &g_var }` address-takes
    // everything it holds: packed constant addresses have no insert
    // site to catch them.
    #[test]
    fn aggregate_constant_store_takes_addresses() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(String::from("cb"), &[], &[I32], Linkage::Internal);
        let global = crate::rvsdg::GlobalId(5);
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                use crate::rvsdg::constant::{ConstantDef, ConstantKind};
                let func_entry = rb.module_tables.constants.intern(ConstantDef {
                    ty: i32_ptr,
                    kind: ConstantKind::FuncAddr(callee),
                });
                let global_entry = rb.module_tables.constants.global_addr(i32_ptr, global);
                // The aggregate's own value type is irrelevant to
                // classification -- the payload KINDS drive the walk.
                let aggregate = rb
                    .module_tables
                    .constants
                    .aggregate(crate::rvsdg::types::I64, &[func_entry, global_entry]);
                let aggregate_value = rb.const_pool_ref(aggregate, crate::rvsdg::types::I64);
                let four = rb.const_i32(4);
                let slot = rb.alloca(crate::rvsdg::types::I64, four, i32_ptr, None);
                rb.store(slot, aggregate_value, None, false);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(facts.address_taken_functions.as_slice(), &[callee]);
        assert_eq!(facts.address_taken_global.as_slice(), &[global]);
    }

    // A function only ever called INDIRECTLY is still externally
    // callable: the callee operand of call_indirect escapes.
    #[test]
    fn indirect_call_callee_is_address_taken() {
        let (mut module, i32_ptr) = test_module();
        let callee = module.declare_fn(String::from("g"), &[], &[I32], Linkage::Internal);
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let function_ptr = rb.func_addr(callee, i32_ptr);
                let signature = rb
                    .module_tables
                    .get_function(callee)
                    .declared_sig
                    .expect("declared functions carry a signature");
                rb.call_indirect(function_ptr, &[], signature);
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(facts.address_taken_functions.as_slice(), &[callee]);
        assert_eq!(facts.calls.len(), 1);
        assert_eq!(facts.calls[0].callee, None);
    }

    // Two pointer-typed results: local_return collapses to Unknown,
    // conservatively, while each return still classifies (param
    // retained, alloca escaped).
    #[test]
    fn multiple_pointer_returns_collapse_to_unknown() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(
            String::from("f"),
            &[i32_ptr],
            &[i32_ptr, i32_ptr],
            Linkage::Internal,
        );
        module
            .define_fn(function, |rb| {
                let param_ptr = rb.param(0);
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                Ok(vec![param_ptr, alloca_a])
            })
            .unwrap();
        let facts = &module.facts[function.0 as usize];
        assert_eq!(facts.local_return, LocalReturn::Unknown);
        assert!(facts.local_captured.is_retained(0));
        assert_eq!(facts.escaped_origins.len(), 1, "the returned alloca");
    }
}
