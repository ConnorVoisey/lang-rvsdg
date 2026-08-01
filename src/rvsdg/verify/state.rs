//! **State edges** -- the compile-time sequencing chain. Every
//! side-effecting node carries a state operand ordering it against the
//! other side-effecting work; today's emitter lowers regions positionally
//! and never reads these edges, so a broken chain is invisible right up
//! until the first pass that TRUSTS state edges (optimisation, control
//! flow reconstruction) turns it into a silent miscompile. This pass is
//! the net under that: it holds state edges to the rule the construction
//! promises.
//!
//! The rule, per region: a node's state operand must be either
//! - the region's `entry_state` -- subregions receive the enclosing chain
//!   through `Region::entry_state` by construction, so this is how a
//!   state edge legitimately crosses a region boundary (state NEVER
//!   travels through region parameters; a State-typed capture parameter
//!   is always a frontend bug), or
//! - an EARLIER state-producing node of the SAME region (side-effecting
//!   nodes are their own output state).
//!
//! Together with the scope pass (which deliberately skips state fields)
//! every operand field of every kind is visited by exactly one pass, so
//! nothing is exempt-by-omission.
//!
//! Every region also carries an `exit_state` closing its chain (checked
//! here against the same rule), so edge-following passes reach subregion
//! state ops without positional scans.
//!
//! One structural limit: a function's `RegionResult` value records the
//! function's final state, but RegionResult values are not members of any
//! region's node list, so their owning region is unrepresented. Their
//! state operand gets the weaker, region-less check at the bottom.

use crate::rvsdg::{
    RegionId, State, ValueId, ValueKind, function_graph::FunctionGraph,
    verify::RVSDGVerificationError,
};

use super::scope::Owner;

impl FunctionGraph {
    pub(super) fn verify_state(&self, owner: &[Owner], errs: &mut Vec<RVSDGVerificationError>) {
        // Side-effecting kinds: the node itself is its output state, so it
        // is a legal source for a later node's state operand.
        let produces_state = |value: ValueId| {
            matches!(
                self.get_value_kind(value),
                ValueKind::Load { .. }
                    | ValueKind::Store { .. }
                    | ValueKind::Alloca { .. }
                    | ValueKind::AtomicLoad { .. }
                    | ValueKind::AtomicStore { .. }
                    | ValueKind::AtomicReadModifyWrite { .. }
                    | ValueKind::CompareAndSwap { .. }
                    | ValueKind::Fence { .. }
                    | ValueKind::Intrinsic { .. }
                    | ValueKind::Call { .. }
                    | ValueKind::CallIndirect { .. }
                    | ValueKind::Gamma { .. }
                    | ValueKind::Theta { .. }
            )
        };

        for (region_index, region) in self.regions.iter().enumerate() {
            let user_region = region_index as u32;

            // The region's exit state closes its chain: entry state for a
            // pure region, else one of its own state-producing nodes.
            // Checked before anything else touches it: INVALID is an
            // out-of-range id, so indexing with it would panic.
            if region.exit_state == State::INVALID {
                errs.push(RVSDGVerificationError::RegionExitStateUnset(RegionId(
                    user_region,
                )));
            } else if region.exit_state != region.entry_state {
                let valid = matches!(
                    owner[region.exit_state.0.0 as usize],
                    Owner::Node { region: r, .. } if r == user_region
                ) && produces_state(region.exit_state.0);
                if !valid {
                    errs.push(RVSDGVerificationError::RegionExitStateInvalid {
                        region: RegionId(user_region),
                        operand: region.exit_state.0,
                    });
                }
            }

            for (position, &user) in region.nodes.iter().enumerate() {
                // Exhaustive over ValueKind: every variant either names its
                // state operand or is listed as pure, so adding a variant
                // forces a decision here.
                let state: State = match self.get_value_kind(user) {
                    ValueKind::Load { state, .. }
                    | ValueKind::Store { state, .. }
                    | ValueKind::Alloca { state, .. }
                    | ValueKind::AtomicLoad { state, .. }
                    | ValueKind::AtomicStore { state, .. }
                    | ValueKind::AtomicReadModifyWrite { state, .. }
                    | ValueKind::CompareAndSwap { state, .. }
                    | ValueKind::Fence { state, .. }
                    | ValueKind::Intrinsic { state, .. }
                    | ValueKind::Call { state, .. }
                    | ValueKind::CallIndirect { state, .. }
                    | ValueKind::Gamma { state, .. }
                    | ValueKind::Theta { state, .. } => *state,
                    // RegionResult is never a region node today (see the
                    // module docs); if one ever is, its state gets the
                    // region-less check below rather than silence.
                    ValueKind::RegionResult { .. } => continue,
                    // Pure values: no state operand.
                    ValueKind::Const(_)
                    | ValueKind::ConstPoolRef(_)
                    | ValueKind::GlobalRef(_)
                    | ValueKind::FuncAddr(_)
                    | ValueKind::Unary { .. }
                    | ValueKind::Binary { .. }
                    | ValueKind::ICmp { .. }
                    | ValueKind::FCmp { .. }
                    | ValueKind::Ternary { .. }
                    | ValueKind::Cast { .. }
                    | ValueKind::ExtractLane { .. }
                    | ValueKind::InsertLane { .. }
                    | ValueKind::ShuffleLanes { .. }
                    | ValueKind::ExtractField { .. }
                    | ValueKind::InsertField { .. }
                    | ValueKind::PtrOffset { .. }
                    | ValueKind::Freeze { .. }
                    | ValueKind::Match { .. }
                    | ValueKind::Project { .. }
                    | ValueKind::RegionParam { .. } => continue,
                };

                if state == region.entry_state {
                    continue;
                }
                match owner[state.0.0 as usize] {
                    Owner::Node {
                        region: r,
                        position: p,
                    } if r == user_region => {
                        if !produces_state(state.0) {
                            errs.push(RVSDGVerificationError::StateEdgeFromNonStateNode {
                                user,
                                operand: state.0,
                            });
                        } else if p >= position as u32 {
                            errs.push(RVSDGVerificationError::StateEdgeUsedBeforeDefinition {
                                user,
                                operand: state.0,
                            });
                        }
                    }
                    _ => errs.push(RVSDGVerificationError::StateEdgeOutOfScope {
                        user,
                        operand: state.0,
                        region: RegionId(user_region),
                    }),
                }
            }
        }

        // RegionResult values (function final states) have no owning-region
        // link, so the strongest sound check is region-less: the final
        // state must be SOME region's entry state or a state-producing
        // node. Wrong-region chains through a valid state node pass here;
        // garbage (a Binary, a dangling id) does not.
        for (index, value) in self.value_kinds.iter().enumerate() {
            if let ValueKind::RegionResult { state, .. } = value {
                let valid =
                    self.regions.iter().any(|r| r.entry_state == *state) || produces_state(state.0);
                if !valid {
                    errs.push(RVSDGVerificationError::StateEdgeFromNonStateNode {
                        user: ValueId(index as u32),
                        operand: state.0,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::rvsdg::{
        ConstValue, Linkage, RVSDGMod, State,
        builder::BranchResult,
        func::FnResult,
        ops::MemoryOrdering,
        types::{BOOL, I32},
        verify::RVSDGVerificationError,
    };

    /// Arm B chains its fence from arm A's fence: a state edge reaching
    /// into a SIBLING region's chain. Emission (positional) would silently
    /// tolerate this today, which is exactly why it must be an error --
    /// the first pass that trusts state edges would reorder across it.
    #[test]
    fn state_edge_into_sibling_arm_is_caught() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        let arm_a_state: Cell<Option<State>> = Cell::new(None);
        rvsdg
            .define_fn(main_fn, |rb, state| {
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let res = rb.gamma(
                    predicate,
                    state,
                    &[],
                    |rb| {
                        // Legal: chains from the arm's entry state.
                        let fenced = rb.fence(state, MemoryOrdering::SequentiallyConsistent);
                        arm_a_state.set(Some(fenced));
                        let zero = rb.const_i32(0);
                        Ok(BranchResult {
                            state: fenced,
                            values: vec![zero],
                        })
                    },
                    |rb| {
                        // Violation: chains from the SIBLING arm's fence.
                        let stolen = arm_a_state.get().expect("first arm runs first");
                        let fenced = rb.fence(stolen, MemoryOrdering::SequentiallyConsistent);
                        let one = rb.const_i32(1);
                        Ok(BranchResult {
                            state: fenced,
                            values: vec![one],
                        })
                    },
                )?;
                Ok(FnResult {
                    state: res.state,
                    values: vec![res.result(0)],
                })
            })
            .unwrap();

        let errs = rvsdg.verify();
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
        let first_fence = Cell::new(None);
        rvsdg
            .define_fn(main_fn, |rb, state| {
                let fenced = rb.fence(state, MemoryOrdering::SequentiallyConsistent);
                first_fence.set(Some(fenced));
                let fenced_again = rb.fence(fenced, MemoryOrdering::SequentiallyConsistent);
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let zero = rb.const_i32(0);
                let res = rb.gamma(
                    predicate,
                    fenced_again,
                    &[],
                    |_rb| {
                        Ok(BranchResult {
                            state: fenced_again,
                            values: vec![zero],
                        })
                    },
                    |_rb| {
                        Ok(BranchResult {
                            state: fenced_again,
                            values: vec![zero],
                        })
                    },
                )?;
                Ok(FnResult {
                    state: res.state,
                    values: vec![res.result(0)],
                })
            })
            .unwrap();

        // Corrupt an arm's exit state to the FUNCTION region's first
        // fence: a state-producing node, but of the wrong region (and
        // not the arm's entry state, which is the second fence).
        let graph = rvsdg.graphs[0].as_mut().unwrap();
        let arm = graph
            .regions
            .iter()
            .position(|region| region.params.is_empty() && region.nodes.is_empty())
            .expect("gamma arm region exists");
        graph.regions[arm].exit_state = first_fence.get().unwrap();

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionExitStateInvalid { .. })),
            "expected an exit-state violation, got: {errs:?}"
        );
    }

    /// Regions are created with `State::INVALID` and every finaliser must
    /// overwrite it; one that slips through must be reported, not chased
    /// through an out-of-range index.
    #[test]
    fn unset_exit_state_is_caught() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb, state| {
                let zero = rb.const_i32(0);
                Ok(FnResult {
                    state,
                    values: vec![zero],
                })
            })
            .unwrap();

        let graph = rvsdg.graphs[0].as_mut().unwrap();
        graph.regions[0].exit_state = State::INVALID;

        let errs = graph.verify(&rvsdg.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionExitStateUnset(_))),
            "expected an unset exit-state error, got: {errs:?}"
        );
    }
}
