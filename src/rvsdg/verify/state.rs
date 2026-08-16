//! **State edges** -- the compile-time sequencing chain. Every
//! side-effecting node carries a state operand ordering it against the
//! other side-effecting work; today's emitter lowers regions positionally
//! and never reads these edges, so a broken chain is invisible right up
//! until the first pass that TRUSTS state edges (optimisation, control
//! flow reconstruction) turns it into a silent miscompile. This pass is
//! the net under that: it holds state edges to the rule the construction
//! promises.
//!
//! Construction threads two chains -- memory and io -- and the rule is
//! the same per chain, per region: a node's state operand must be either
//! - the chain's entry (the matching state-params tail entry, order
//!   [memory, io]) -- subregions receive the enclosing chains through
//!   their tails by construction, so this is how a state edge
//!   legitimately crosses a region boundary (state NEVER travels through
//!   region parameters; a State-typed capture parameter is always a
//!   frontend bug), or
//! - an EARLIER node of the SAME region producing on that chain
//!   (side-effecting nodes are their own output state; the io producers
//!   are the calls and the constructs' io state projections).
//!
//! Gamma and theta carry no state operands: their chain inputs are their
//! subregions' entry tails (identical across arms, checked here), and
//! their chain outputs are their state projections, held to the same
//! per-chain rule wherever they are consumed.
//!
//! Together with the scope pass (which deliberately skips state fields)
//! every operand field of every kind is visited by exactly one pass, so
//! nothing is exempt-by-omission.
//!
//! Every region's state-results tail closes both chains (checked here
//! against the same rule), so edge-following passes reach subregion
//! state ops without positional scans.

use crate::rvsdg::{
    RegionId, State, StateKind, ValueId, ValueKind, function_graph::FunctionGraph, types::TypeRef,
    verify::RVSDGVerificationError,
};

use super::scope::Owner;

impl FunctionGraph {
    pub(super) fn verify_state(&self, owner: &[Owner], errs: &mut Vec<RVSDGVerificationError>) {
        // Side-effecting values: the node itself is its output state, so
        // it is a legal source for a later node's state operand. Merges
        // are ordering structure over reads, legitimately consumed by a
        // write or the exit. A construct is NOT a state value -- its
        // chains continue on its state projections, recognised by their
        // State type (which the typing pass holds honest).
        let produces_state = |value: ValueId| {
            let kind = self.get_value_kind(value);
            kind.is_memory_op()
                || kind.is_call()
                || matches!(kind, ValueKind::StateMerge { .. })
                || (matches!(kind, ValueKind::Project { .. })
                    && matches!(
                        self.get_value_type(value),
                        TypeRef::State(StateKind::MemoryRead(_) | StateKind::MemoryWrite(_))
                    ))
        };

        // Io producers: the values that advance the io chain (a legal
        // source for an io_state operand or the io exit slot).
        let produces_io = |value: ValueId| {
            let kind = self.get_value_kind(value);
            kind.is_call()
                || (matches!(kind, ValueKind::Project { .. })
                    && matches!(
                        self.get_value_type(value),
                        TypeRef::State(StateKind::InputOutput)
                    ))
        };

        for region_index in 0..self.regions.len() {
            let user_region = region_index as u32;

            // Each chain's exit closes it: the chain's entry for a pure
            // region, else one of the region's own producers on that
            // chain. Tail order is [memory, io] on both sides.
            let entries = self.region_state_params(RegionId(user_region));
            let entry_state = entries.first().copied();
            let entry_io = entries.get(1).copied();
            let exits = self.region_state_results(RegionId(user_region));
            for (exit, entry, chain_producer) in [
                (
                    exits.first().copied(),
                    entry_state,
                    &produces_state as &dyn Fn(ValueId) -> bool,
                ),
                (exits.get(1).copied(), entry_io, &produces_io),
            ] {
                match exit {
                    None => errs.push(RVSDGVerificationError::RegionExitStateUnset(RegionId(
                        user_region,
                    ))),
                    Some(exit) if Some(exit) != entry => {
                        let valid = matches!(
                            owner[exit.0 as usize],
                            Owner::Node { region: r, .. } if r == user_region
                        ) && chain_producer(exit);
                        if !valid {
                            errs.push(RVSDGVerificationError::RegionExitStateInvalid {
                                region: RegionId(user_region),
                                operand: exit,
                            });
                        }
                    }
                    Some(_) => {}
                }
            }

            for (position, &user) in self.region_nodes(RegionId(user_region)).iter().enumerate() {
                // Which chains a kind consumes through operands is
                // defined once, exhaustively, by
                // ValueKind::memory_state_operand / io_state_operand.
                // Constructs are the exception handled here: their chain
                // inputs are their subregions' entry tails -- every
                // subregion must carry the SAME parent-side values
                // (checked here), each a legal chain value of this
                // region before the construct (checked below through
                // the first subregion's entries). Merge inputs are
                // checked as ordinary operands, not as a state slot.
                let kind = self.get_value_kind(user);
                let (state, io): (State, Option<State>) = if kind.is_construct() {
                    let subregions = self.construct_subregions(user);
                    let first = self.region_state_params(subregions[0]);
                    for &sub in &subregions[1..] {
                        if self.region_state_params(sub) != first {
                            errs.push(RVSDGVerificationError::ConstructEntryTailsDisagree {
                                construct: user,
                                region: sub,
                            });
                        }
                    }
                    let (Some(&memory), Some(&io)) = (first.first(), first.get(1)) else {
                        // Arity errors are the typing pass's report.
                        continue;
                    };
                    (State(memory), Some(State(io)))
                } else {
                    match kind.memory_state_operand() {
                        Some(state) => (state, kind.io_state_operand()),
                        None => continue,
                    }
                };

                let mut check_chain =
                    |operand: State,
                     entry: Option<ValueId>,
                     chain_producer: &dyn Fn(ValueId) -> bool| {
                        if Some(operand.0) == entry {
                            return;
                        }
                        match owner[operand.0.0 as usize] {
                            Owner::Node {
                                region: r,
                                position: p,
                            } if r == user_region => {
                                if !chain_producer(operand.0) {
                                    errs.push(RVSDGVerificationError::StateEdgeFromNonStateNode {
                                        user,
                                        operand: operand.0,
                                    });
                                } else if p >= position as u32 {
                                    errs.push(
                                        RVSDGVerificationError::StateEdgeUsedBeforeDefinition {
                                            user,
                                            operand: operand.0,
                                        },
                                    );
                                }
                            }
                            _ => errs.push(RVSDGVerificationError::StateEdgeOutOfScope {
                                user,
                                operand: operand.0,
                                region: RegionId(user_region),
                            }),
                        }
                    };

                check_chain(state, entry_state, &produces_state);
                if let Some(io) = io {
                    check_chain(io, entry_io, &produces_io);
                }
            }
        }
    }

    /// **Chain continuity** -- the complement of the edge rules above.
    /// Those check that every state edge points backwards at a legal
    /// producer; this checks the forward direction: every effectful
    /// producer must be transitively CONSUMED into some region's exit
    /// tail, or its ordering has been lost and dead node elimination
    /// will silently delete a real effect. This is the net under the
    /// builder's seeding/assembly contract and under any pass that
    /// rewrites chains (passthrough reroute).
    ///
    /// Mechanically: walk each chain backwards from every region's exit
    /// tails (memory and io separately; a construct's state projection
    /// continues into its subregions' entry tail for that chain), then
    /// require every memory op, merge and call to be memory-reached,
    /// calls additionally io-reached, and each construct whose
    /// subregions do NOT pass a chain through to have that chain's
    /// projection reached. A fully bypassed pure construct legitimately
    /// sits off-chain.
    pub(super) fn verify_chain_continuity(&self, errs: &mut Vec<RVSDGVerificationError>) {
        let mut memory_reached = vec![false; self.value_kinds.len()];
        let mut io_reached = vec![false; self.value_kinds.len()];
        let mut stack: Vec<ValueId> = Vec::new();

        // The memory chain's predecessors of a reached value.
        let push_memory = |value: ValueId, stack: &mut Vec<ValueId>| {
            let kind = self.get_value_kind(value);
            if let Some(state) = kind.memory_state_operand() {
                stack.push(state.0);
                return;
            }
            match kind {
                ValueKind::StateMerge { inputs } => {
                    stack.extend_from_slice(self.value_pool.get(*inputs));
                }
                // A construct's memory projection continues into the
                // subregions' entry tail (identical across arms).
                ValueKind::Project { call, .. }
                    if matches!(
                        self.get_value_type(value),
                        TypeRef::State(StateKind::MemoryRead(_) | StateKind::MemoryWrite(_))
                    ) =>
                {
                    if let Some(&sub) = self.construct_subregions(*call).first()
                        && let Some(&entry) = self.region_state_params(sub).first()
                    {
                        stack.push(entry);
                    }
                }
                _ => {}
            }
        };
        let push_io = |value: ValueId, stack: &mut Vec<ValueId>| {
            let kind = self.get_value_kind(value);
            if let Some(io) = kind.io_state_operand() {
                stack.push(io.0);
                return;
            }
            if let ValueKind::Project { call, .. } = kind
                && matches!(
                    self.get_value_type(value),
                    TypeRef::State(StateKind::InputOutput)
                )
                && let Some(&sub) = self.construct_subregions(*call).first()
                && let Some(&entry) = self.region_state_params(sub).get(1)
            {
                stack.push(entry);
            }
        };

        for (slot, reached, push) in [
            (
                0usize,
                &mut memory_reached,
                &push_memory as &dyn Fn(ValueId, &mut Vec<ValueId>),
            ),
            (1usize, &mut io_reached, &push_io),
        ] {
            stack.clear();
            for region_index in 0..self.regions.len() {
                let region = RegionId(region_index as u32);
                let exit = self.region_state_results(region).get(slot).copied();
                let entry = self.region_state_params(region).get(slot).copied();
                // A passed-through slot (exit == entry) roots nothing: the
                // entry is a parent-side value whose consumption is the
                // parent's own responsibility. Rooting it here would let a
                // pure subregion's tails mask a parent exit chain that
                // skips the values they happen to name.
                if let Some(exit) = exit
                    && Some(exit) != entry
                {
                    stack.push(exit);
                }
            }
            while let Some(value) = stack.pop() {
                if reached[value.0 as usize] {
                    continue;
                }
                reached[value.0 as usize] = true;
                push(value, &mut stack);
            }
        }

        for region_index in 0..self.regions.len() {
            for &node in self.region_nodes(RegionId(region_index as u32)) {
                let kind = self.get_value_kind(node);
                if kind.is_memory_op()
                    || kind.is_call()
                    || matches!(kind, ValueKind::StateMerge { .. })
                {
                    if !memory_reached[node.0 as usize] {
                        errs.push(RVSDGVerificationError::StateEffectUnrooted {
                            value: node,
                            chain: "memory",
                        });
                    }
                    if kind.is_call() && !io_reached[node.0 as usize] {
                        errs.push(RVSDGVerificationError::StateEffectUnrooted {
                            value: node,
                            chain: "io",
                        });
                    }
                }
                match kind {
                    ValueKind::Gamma { .. } | ValueKind::Theta { .. } => {
                        let projections = self.construct_state_projections(node);
                        for (slot, chain, reached, projection) in [
                            (0usize, "memory", &memory_reached, projections.memory),
                            (1usize, "io", &io_reached, projections.io),
                        ] {
                            let impure = self.construct_subregions(node).iter().any(|&sub| {
                                self.region_state_results(sub).get(slot)
                                    != self.region_state_params(sub).get(slot)
                            });
                            if !impure {
                                continue;
                            }
                            // An impure chain must flow through the
                            // construct: its projection exists and is
                            // consumed.
                            let rooted =
                                projection.is_some_and(|projection| reached[projection.0 as usize]);
                            if !rooted {
                                errs.push(RVSDGVerificationError::StateEffectUnrooted {
                                    value: projection.unwrap_or(node),
                                    chain,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ConstValue, Linkage, RVSDGMod, State, ValueId, ValueKind,
        function_graph::FunctionGraph,
        ops::MemoryOrdering,
        types::{BOOL, I32},
        verify::RVSDGVerificationError,
    };

    /// The fence nodes of a graph in creation (id) order. The builder no
    /// longer hands out node ids for state ops, so corruption tests find
    /// their targets by kind.
    fn fence_nodes(graph: &FunctionGraph) -> Vec<ValueId> {
        (0..graph.value_kinds.len() as u32)
            .map(ValueId)
            .filter(|&id| matches!(graph.get_value_kind(id), ValueKind::Fence { .. }))
            .collect()
    }

    /// Arm B chains its fence from arm A's fence: a state edge reaching
    /// into a SIBLING region's chain. Emission (positional) would silently
    /// tolerate this today, which is exactly why it must be an error --
    /// the first pass that trusts state edges would reorder across it.
    /// The builder threads state internally and cannot produce this, so
    /// the corruption is applied after construction.
    #[test]
    fn state_edge_into_sibling_arm_is_caught() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb| {
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let res = rb.gamma(
                    predicate,
                    &[],
                    |rb| {
                        rb.fence(MemoryOrdering::SequentiallyConsistent);
                        Ok(vec![rb.const_i32(0)])
                    },
                    |rb| {
                        rb.fence(MemoryOrdering::SequentiallyConsistent);
                        Ok(vec![rb.const_i32(1)])
                    },
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        // Redirect arm B's fence to chain from arm A's fence.
        let graph = rvsdg.graphs[0].as_mut().unwrap();
        let [arm_a_fence, arm_b_fence] = fence_nodes(graph)[..] else {
            panic!("expected exactly two fences");
        };
        match &mut graph.value_kinds[arm_b_fence.0 as usize] {
            ValueKind::Fence { state, .. } => *state = State(arm_a_fence),
            other => panic!("expected a fence, got {other:?}"),
        }

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::StateEdgeOutOfScope { .. })),
            "expected a state scope violation, got: {errs:?}"
        );
    }

    /// A region whose exit state points at another region's chain (here:
    /// corrupted after construction) must be rejected.
    #[test]
    fn foreign_exit_state_is_caught() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb| {
                rb.fence(MemoryOrdering::SequentiallyConsistent);
                rb.fence(MemoryOrdering::SequentiallyConsistent);
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let zero = rb.const_i32(0);
                let res = rb.gamma(predicate, &[], |_rb| Ok(vec![zero]), |_rb| Ok(vec![zero]))?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        // Corrupt an arm's exit state to the FUNCTION region's first
        // fence: a state-producing node, but of the wrong region (and
        // not the arm's entry state, which is the second fence).
        let graph = rvsdg.graphs[0].as_mut().unwrap();
        let first_fence = fence_nodes(graph)[0];
        let arm = graph
            .regions
            .iter()
            .position(|region| region.params_len == 0 && region.nodes_len == 0)
            .expect("gamma arm region exists");
        let arm = graph.regions[arm].clone();
        arm.state_results_mut(&mut graph.value_pool)[0] = first_fence;

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionExitStateInvalid { .. })),
            "expected an exit-state violation, got: {errs:?}"
        );
    }

    /// An effectful producer the exit chain skips: the exit tail is
    /// corrupted back to the entry state, claiming a pure region while a
    /// store sits inside it. Edge rules alone accept this (exit ==
    /// entry is the legal pure shape); continuity must reject it, since
    /// elimination would delete the store.
    #[test]
    fn store_skipped_by_exit_chain_is_caught() {
        use crate::rvsdg::types::{PtrType, TypeRef};
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr = TypeRef::Ptr(rvsdg.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        }));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[I32], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb| {
                let x = rb.param(0);
                let one = rb.const_i32(1);
                let slot = rb.alloca(I32, one, ptr, None);
                rb.store(slot, x, None, false);
                Ok(vec![x])
            })
            .unwrap();

        // Point the body's memory exit back at its entry state param.
        let graph = rvsdg.graphs[0].as_mut().unwrap();
        let entry = graph.region_state_params(crate::rvsdg::RegionId(0))[0];
        let body = graph.regions[0].clone();
        body.state_results_mut(&mut graph.value_pool)[0] = entry;

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                RVSDGVerificationError::StateEffectUnrooted {
                    chain: "memory",
                    ..
                }
            )),
            "expected an unrooted-effect error, got: {errs:?}"
        );
    }

    /// The masking variant of the skipped-store shape: a pure gamma sits
    /// after the store, and its (passed-through) arm tails name the
    /// store. Those tails must not root the store -- with the function's
    /// exit corrupted back to its entry, the store is still off the
    /// parent chain and must be reported.
    #[test]
    fn store_skipped_behind_pure_construct_is_caught() {
        use crate::rvsdg::types::{PtrType, TypeRef};
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr = TypeRef::Ptr(rvsdg.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        }));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[I32], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb| {
                let x = rb.param(0);
                let one = rb.const_i32(1);
                let slot = rb.alloca(I32, one, ptr, None);
                rb.store(slot, x, None, false);
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let res = rb.gamma(
                    predicate,
                    &[x],
                    |rb| Ok(vec![rb.param(0)]),
                    |rb| Ok(vec![rb.param(0)]),
                )?;
                Ok(vec![res.result(0)])
            })
            .unwrap();

        let graph = rvsdg.graphs[0].as_mut().unwrap();
        let entry = graph.region_state_params(crate::rvsdg::RegionId(0))[0];
        let body = graph.regions[0].clone();
        body.state_results_mut(&mut graph.value_pool)[0] = entry;

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                RVSDGVerificationError::StateEffectUnrooted {
                    chain: "memory",
                    ..
                }
            )),
            "expected an unrooted-effect error, got: {errs:?}"
        );
    }

    /// A region whose state-results tail is missing an entry (corrupted
    /// here after construction) must be reported, not chased through an
    /// out-of-range index.
    #[test]
    fn unset_exit_state_is_caught() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb| Ok(vec![rb.const_i32(0)]))
            .unwrap();

        let graph = rvsdg.graphs[0].as_mut().unwrap();
        graph.regions[0].state_results_len = 0;

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionExitStateUnset(_))),
            "expected an unset exit-state error, got: {errs:?}"
        );
    }
}
