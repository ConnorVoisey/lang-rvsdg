//! Origin resolution: compress every `Derived` link and join to a
//! final origin, in place, at the end of construction.
//!
//! The origin graph is structurally a backward DAG almost everywhere:
//! builders only ever link a derived pointer to an ALREADY-CREATED
//! value, so every slot edge points at a lower id, and one ascending
//! in-place pass resolves it -- a target is final before any consumer
//! reads it. The only source of forward edges (and therefore cycles)
//! is join events, and of those only a theta parameter's recirculation
//! actually points forward. Values with joins, or depending on one,
//! are DEFERRED to a strongly-connected-component pass over just that
//! small set: every member of a component gets the component-wide
//! answer -- the named origins entering the component if they agree,
//! otherwise Unknown -- and a widening component fires an escape for
//! every named origin it discards, because a store through the widened
//! pointer lands in class 0 and the discarded allocation must live
//! there to conflict with it. First-visit memoization without the SCC
//! treatment would be order-dependent exactly there: the recirculation
//! reaches the parameter itself.

use crate::rvsdg::{ValueId, function_graph::FunctionGraph, memory_alias::origin::MemoryOrigin};

/// Sentinel for "not yet visited" in the Tarjan order array.
const UNVISITED: u32 = u32::MAX;

/// The resolver's per-function work buffers, recycled across functions
/// on the construction scratch: the deferred list and the Tarjan walk
/// state. Every field is re-initialised at use, so nothing here carries
/// meaning between functions.
#[derive(Debug, Default)]
pub(crate) struct ResolveScratch {
    /// Values the forward scan could not finalize: join-involved, or
    /// depending on a deferred target. Ascending, so dense membership
    /// is one binary search.
    deferred: Vec<u32>,
    visit_order: Vec<u32>,
    low_link: Vec<u32>,
    on_stack: Vec<bool>,
    component_stack: Vec<u32>,
    /// The DFS call stack, explicit: (dense node, next edge).
    walk_frames: Vec<(u32, u32)>,
    /// Escapes fired by widening components, applied at the end.
    widened_escapes: Vec<ValueId>,
    /// The named external sources of the component being resolved.
    named_targets: Vec<ValueId>,
}

impl FunctionGraph {
    /// Compress every `Derived` tag to its final origin. Called once at
    /// the end of construction, before event classification; consumes
    /// the join events (their buffer and the resolver's own work
    /// buffers go back to the scratch emptied, keeping their capacity
    /// warm). After this returns, no `Derived` remains -- later readers
    /// never chase.
    pub(crate) fn resolve_origins(&mut self) {
        let mut joins = std::mem::take(&mut self.mem_facts.join_events);
        joins.sort_unstable_by_key(|(join, _)| join.0);
        let mut scratch = std::mem::take(&mut self.mem_facts.resolve);
        scratch.deferred.clear();

        // Phase 1: the ascending forward scan that handles the acyclic
        // majority: a Derived value with no joins copies its slot
        // target's already-final origin in place. Values with join
        // events, or whose target is itself deferred (still Derived
        // when read), wait for the component pass.
        let mut join_cursor = 0usize;
        for value in 0..self.memory_origins.len() as u32 {
            while join_cursor < joins.len() && joins[join_cursor].0.0 < value {
                join_cursor += 1;
            }
            let has_joins = joins
                .get(join_cursor)
                .is_some_and(|(join, _)| join.0 == value);
            let MemoryOrigin::Derived(target) = self.memory_origins[value as usize] else {
                debug_assert!(
                    !has_joins,
                    "join event on a value without a Derived slot link"
                );
                continue;
            };
            debug_assert!(
                target.0 < value,
                "slot edges are structurally backward: builders link to existing values"
            );
            if has_joins {
                scratch.deferred.push(value);
                continue;
            }
            match self.memory_origins[target.0 as usize] {
                // The target waits on a join component; so do we.
                MemoryOrigin::Derived(_) => scratch.deferred.push(value),
                // A pointer chased into a non-pointer is a builder bug;
                // widen in release.
                MemoryOrigin::None => {
                    debug_assert!(false, "Derived link to a None-origin value");
                    self.memory_origins[value as usize] = MemoryOrigin::Unknown;
                }
                origin => self.memory_origins[value as usize] = origin,
            }
        }

        if !scratch.deferred.is_empty() {
            self.resolve_deferred_components(&mut scratch, &joins);
        }
        self.mem_facts.resolve = scratch;
        // Consumed: a second resolution must not re-apply stale joins
        // to already-compressed origins.
        joins.clear();
        self.mem_facts.join_events = joins;

        debug_assert!(
            !self
                .memory_origins
                .iter()
                .any(|origin| matches!(origin, MemoryOrigin::Derived(_))),
            "Derived origin survived resolution"
        );
    }

    /// The component pass over the deferred set (join-involved values
    /// and their dependents): iterative Tarjan on DENSE indices into
    /// `deferred`, which is ascending, so membership is one binary
    /// search. Final origins are written in place at each component pop
    /// -- pops are reverse-topological, so a popped member is final
    /// before anything outside its component reads it, and phase 1
    /// already finalized every non-deferred target.
    fn resolve_deferred_components(
        &mut self,
        scratch: &mut ResolveScratch,
        joins: &[(ValueId, ValueId)],
    ) {
        // Destructure for disjoint field borrows; the buffers keep
        // their capacity from previous functions.
        let ResolveScratch {
            deferred,
            visit_order,
            low_link,
            on_stack,
            component_stack,
            walk_frames,
            widened_escapes,
            named_targets,
        } = scratch;
        let deferred: &[u32] = deferred;
        let node_count = deferred.len();
        visit_order.clear();
        visit_order.resize(node_count, UNVISITED);
        low_link.clear();
        low_link.resize(node_count, 0);
        on_stack.clear();
        on_stack.resize(node_count, false);
        component_stack.clear();
        walk_frames.clear();
        widened_escapes.clear();
        let mut next_visit_order = 0u32;

        // Edge `edge_index` of a deferred value: 0 is the Derived slot
        // link (still present -- deferred values are untouched until
        // their component pops), 1.. its join events.
        let edge_target = |origins: &[MemoryOrigin], value: u32, edge_index: u32| -> Option<u32> {
            if edge_index == 0 {
                match origins[value as usize] {
                    MemoryOrigin::Derived(target) => Some(target.0),
                    _ => unreachable!("deferred value lost its Derived slot before its pop"),
                }
            } else {
                let start = joins.partition_point(|(join, _)| join.0 < value);
                let end = joins[start..].partition_point(|(join, _)| join.0 == value) + start;
                joins[start..end]
                    .get(edge_index as usize - 1)
                    .map(|(_, target)| target.0)
            }
        };

        for dense_root in 0..node_count as u32 {
            if visit_order[dense_root as usize] != UNVISITED {
                continue;
            }
            walk_frames.push((dense_root, 0));
            visit_order[dense_root as usize] = next_visit_order;
            low_link[dense_root as usize] = next_visit_order;
            next_visit_order += 1;
            component_stack.push(dense_root);
            on_stack[dense_root as usize] = true;

            while let Some(&mut (dense_node, ref mut next_edge)) = walk_frames.last_mut() {
                let value = deferred[dense_node as usize];
                if let Some(target) = edge_target(&self.memory_origins, value, *next_edge) {
                    *next_edge += 1;
                    // Only deferred targets are graph nodes; everything
                    // else is a final leaf consumed at resolution.
                    let Ok(dense_target) = deferred.binary_search(&target) else {
                        continue;
                    };
                    let dense_target = dense_target as u32;
                    if visit_order[dense_target as usize] == UNVISITED {
                        walk_frames.push((dense_target, 0));
                        visit_order[dense_target as usize] = next_visit_order;
                        low_link[dense_target as usize] = next_visit_order;
                        next_visit_order += 1;
                        component_stack.push(dense_target);
                        on_stack[dense_target as usize] = true;
                    } else if on_stack[dense_target as usize] {
                        low_link[dense_node as usize] =
                            low_link[dense_node as usize].min(visit_order[dense_target as usize]);
                    }
                    continue;
                }
                // Edges exhausted: pop the frame, fold the lowlink into
                // the parent, resolve the component if this roots one.
                walk_frames.pop();
                if let Some(&mut (parent, _)) = walk_frames.last_mut() {
                    low_link[parent as usize] =
                        low_link[parent as usize].min(low_link[dense_node as usize]);
                }
                if low_link[dense_node as usize] != visit_order[dense_node as usize] {
                    continue;
                }
                let first_member = component_stack
                    .iter()
                    .rposition(|&member| member == dense_node)
                    .expect("component root must be on the stack");
                let members = &component_stack[first_member..];

                // Gather the component's external sources: the final
                // origins of every out-of-component target. Agreement =
                // exactly one distinct named origin and no Unknown.
                let mut named: Option<MemoryOrigin> = None;
                named_targets.clear();
                let mut disagree = false;
                let mut saw_unknown = false;
                for &member in members {
                    let member_value = deferred[member as usize];
                    for edge_index in 0.. {
                        let Some(target) =
                            edge_target(&self.memory_origins, member_value, edge_index)
                        else {
                            break;
                        };
                        if let Ok(dense_target) = deferred.binary_search(&target) {
                            if on_stack[dense_target] {
                                continue; // in-component edge
                            }
                        }
                        match self.memory_origins[target as usize] {
                            MemoryOrigin::Unknown => saw_unknown = true,
                            // Off-component deferred targets popped
                            // first (reverse-topological); phase 1
                            // finalized everything else.
                            MemoryOrigin::Derived(_) => {
                                unreachable!("out-of-component target left Derived")
                            }
                            MemoryOrigin::None => {
                                debug_assert!(false, "Derived link to a None-origin value");
                                saw_unknown = true;
                            }
                            origin => {
                                if named.is_none() {
                                    named = Some(origin);
                                } else if named != Some(origin) {
                                    disagree = true;
                                }
                                named_targets.push(ValueId(target));
                            }
                        }
                    }
                }

                let component_origin = match (named, disagree || saw_unknown) {
                    (Some(origin), false) => origin,
                    // Disagreement or an Unknown source widens; every
                    // named origin the component saw must escape.
                    (Some(_), true) => {
                        widened_escapes.append(named_targets);
                        MemoryOrigin::Unknown
                    }
                    // No external source at all (a closed cycle nothing
                    // feeds): unreachable pointers, Unknown is sound.
                    (None, _) => MemoryOrigin::Unknown,
                };
                for &member in members {
                    self.memory_origins[deferred[member as usize] as usize] = component_origin;
                }
                for &member in members {
                    on_stack[member as usize] = false;
                }
                component_stack.truncate(first_member);
            }
        }

        for &value in widened_escapes.iter() {
            self.record_escape_event(value);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::rvsdg::{
        ICmpPred, Linkage, RVSDGMod,
        builder::LoopResult,
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

    // int f() { int arr[4]; ... arr[1] ... arr[1][1]... }
    // A chain of pointer offsets compresses to the alloca in the
    // forward scan; nothing is deferred.
    #[test]
    fn offset_chain_compresses_to_alloca() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let array_ptr = rb.alloca(I32, four, i32_ptr, None);
                let one = rb.const_i64(1);
                let first_offset = rb.ptr_offset(array_ptr, I32, &[one], i32_ptr, true);
                let second_offset = rb.ptr_offset(first_offset, I32, &[one], i32_ptr, true);

                rb.graph.resolve_origins();

                let alloca_origin = rb.graph.get_memory_origin(array_ptr);
                assert!(matches!(alloca_origin, MemoryOrigin::Alloca(_)));
                assert_eq!(rb.graph.get_memory_origin(first_offset), alloca_origin);
                assert_eq!(rb.graph.get_memory_origin(second_offset), alloca_origin);
                assert!(rb.graph.mem_facts.escape_events.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // int f() { int arr[4]; for (p = arr; ...; p++) ...; }
    // The recirculating loop pointer is a cycle through the theta
    // parameter; its only external source is the alloca, so the whole
    // component agrees on Alloca and nothing escapes. This is the case
    // the deferred SCC pass exists for.
    #[test]
    fn theta_recirculation_agrees_on_alloca() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let array_ptr = rb.alloca(I32, four, i32_ptr, None);
                let loop_results = rb.theta(&[array_ptr], |body| {
                    let cursor = body.param(0);
                    let one = body.const_i64(1);
                    let advanced = body.ptr_offset(cursor, I32, &[one], i32_ptr, true);
                    let zero = body.const_i32(0);
                    let repeat = body.icmp(ICmpPred::Eq, zero, zero);
                    let condition = body.bool_predicate(repeat);
                    Ok(LoopResult {
                        condition,
                        next_vars: vec![advanced],
                    })
                })?;

                rb.graph.resolve_origins();

                let alloca_origin = rb.graph.get_memory_origin(array_ptr);
                assert!(matches!(alloca_origin, MemoryOrigin::Alloca(_)));
                assert_eq!(
                    rb.graph.get_memory_origin(loop_results.result(0)),
                    alloca_origin,
                    "the loop-carried pointer resolves through the cycle to the alloca"
                );
                assert!(rb.graph.mem_facts.escape_events.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // int f(c) { int a[4], b[4]; int *q = c ? a : b; }  (as a gamma)
    // Two distinct named origins disagree: the projection widens to
    // Unknown and BOTH discarded allocas escape.
    #[test]
    fn gamma_disagreement_widens_and_escapes_both() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let alloca_b = rb.alloca(I32, four, i32_ptr, None);
                let zero = rb.const_i32(0);
                let flag = rb.icmp(ICmpPred::Eq, zero, zero);
                let predicate = rb.bool_predicate(flag);
                let gamma_results = rb.gamma(
                    predicate,
                    &[alloca_a, alloca_b],
                    |arm| Ok(vec![arm.param(0)]),
                    |arm| Ok(vec![arm.param(1)]),
                )?;

                rb.graph.resolve_origins();

                assert_eq!(
                    rb.graph.get_memory_origin(gamma_results.result(0)),
                    MemoryOrigin::Unknown
                );
                let escaped_origins: Vec<MemoryOrigin> = rb
                    .graph
                    .mem_facts
                    .escape_events
                    .iter()
                    .map(|&value| rb.graph.get_memory_origin(value))
                    .collect();
                assert_eq!(escaped_origins.len(), 2);
                assert!(escaped_origins.contains(&rb.graph.get_memory_origin(alloca_a)));
                assert!(escaped_origins.contains(&rb.graph.get_memory_origin(alloca_b)));
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // Both arms return the SAME allocation: sources agree, the
    // projection keeps the named origin, nothing escapes.
    #[test]
    fn gamma_agreement_keeps_named_origin() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let array_ptr = rb.alloca(I32, four, i32_ptr, None);
                let zero = rb.const_i32(0);
                let flag = rb.icmp(ICmpPred::Eq, zero, zero);
                let predicate = rb.bool_predicate(flag);
                let one = rb.const_i64(1);
                let gamma_results = rb.gamma(
                    predicate,
                    &[array_ptr],
                    |arm| Ok(vec![arm.param(0)]),
                    // The false arm returns an OFFSET of the same
                    // allocation: still the same origin.
                    |arm| {
                        let offset = arm.ptr_offset(arm.param(0), I32, &[one], i32_ptr, true);
                        Ok(vec![offset])
                    },
                )?;

                rb.graph.resolve_origins();

                assert_eq!(
                    rb.graph.get_memory_origin(gamma_results.result(0)),
                    rb.graph.get_memory_origin(array_ptr)
                );
                assert!(rb.graph.mem_facts.escape_events.is_empty());
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }

    // A select between a named origin and a LOADED pointer: Unknown is
    // among the sources, so the component widens and the named side
    // escapes (the loaded side has no origin to discard).
    #[test]
    fn select_with_unknown_source_widens_and_escapes_named() {
        let (mut module, i32_ptr) = test_module();
        let function = module.declare_fn(String::from("f"), &[], &[I32], Linkage::Internal);
        module
            .define_fn(function, |rb| {
                let four = rb.const_i32(4);
                let alloca_a = rb.alloca(I32, four, i32_ptr, None);
                let slot = rb.alloca(i32_ptr, four, i32_ptr, None);
                let loaded_ptr = rb.load(slot, i32_ptr, None, false);
                let zero = rb.const_i32(0);
                let flag = rb.icmp(ICmpPred::Eq, zero, zero);
                let selected = rb.ternary(flag, alloca_a, loaded_ptr, i32_ptr);

                rb.graph.resolve_origins();

                assert_eq!(rb.graph.get_memory_origin(selected), MemoryOrigin::Unknown);
                let escaped_origins: Vec<MemoryOrigin> = rb
                    .graph
                    .mem_facts
                    .escape_events
                    .iter()
                    .map(|&value| rb.graph.get_memory_origin(value))
                    .collect();
                assert_eq!(escaped_origins.len(), 1);
                assert_eq!(escaped_origins[0], rb.graph.get_memory_origin(alloca_a));
                Ok(vec![rb.const_i32(0)])
            })
            .unwrap();
    }
}
