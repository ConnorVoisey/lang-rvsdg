//! **Predicate continuation form** (Bahmann, Reissmann, Jahre, Meyer 2015,
//! Definition 2.6): the RVSDG normal form that perfect control-flow
//! reconstruction requires. Every predicate-defining node must have at most
//! one consumer, that consumer must be a gamma decision, a theta repetition
//! predicate, or a region result, and no node may have two
//! predicate-defining predecessors.
//!
//! In this codebase the only predicate-defining nodes are `Match`
//! conversions (auxiliary selectors are integer values; every demux, branch
//! and theta emits its own match adjacent to itself -- see the emitter). A
//! match may therefore only ever flow into exactly the three places the
//! definition allows, and because matches never feed anything else, the
//! "two predicate-defining predecessors" clause holds as a corollary: a
//! node's operands can contain at most one match, its condition.

use rustc_hash::FxHashMap;

use crate::rvsdg::{
    RegionId, ValueId, ValueKind, function_graph::FunctionGraph, verify::RVSDGVerificationError,
};

/// How one match value is consumed.
#[derive(Default)]
struct MatchUses {
    /// Legal consumptions: gamma conditions, theta conditions, region
    /// results.
    predicate_slots: u32,
    /// Any other reference (an operand of an ordinary node, a gamma input,
    /// a theta loop variable, ...): always a violation.
    other: u32,
}

impl FunctionGraph {
    pub(super) fn verify_predicate_form(&self, errs: &mut Vec<RVSDGVerificationError>) {
        // The match values, each with a use tally.
        let mut matches: FxHashMap<ValueId, MatchUses> = FxHashMap::default();
        for (index, value) in self.value_kinds.iter().enumerate() {
            if matches!(value, ValueKind::Match { .. }) {
                matches.insert(ValueId(index as u32), MatchUses::default());
            }
        }
        if matches.is_empty() {
            return;
        }

        let tally =
            |id: ValueId, predicate_slot: bool, matches: &mut FxHashMap<ValueId, MatchUses>| {
                if let Some(uses) = matches.get_mut(&id) {
                    if predicate_slot {
                        uses.predicate_slots += 1;
                    } else {
                        uses.other += 1;
                    }
                }
            };

        for value in &self.value_kinds {
            match &value {
                // The two legal condition slots.
                ValueKind::Gamma {
                    condition, inputs, ..
                } => {
                    tally(*condition, true, &mut matches);
                    for &input in self.value_pool.get(*inputs) {
                        tally(input, false, &mut matches);
                    }
                }
                ValueKind::Theta {
                    condition,
                    loop_vars,
                    ..
                } => {
                    tally(*condition, true, &mut matches);
                    for &var in self.value_pool.get(*loop_vars) {
                        tally(var, false, &mut matches);
                    }
                }

                // Everything else: every value operand is an "other" use.
                ValueKind::Unary { operand, .. } | ValueKind::Cast { value: operand, .. } => {
                    tally(*operand, false, &mut matches);
                }
                ValueKind::Binary { left, right, .. }
                | ValueKind::ICmp { left, right, .. }
                | ValueKind::FCmp { left, right, .. } => {
                    tally(*left, false, &mut matches);
                    tally(*right, false, &mut matches);
                }
                ValueKind::Ternary {
                    condition,
                    true_val,
                    false_val,
                } => {
                    tally(*condition, false, &mut matches);
                    tally(*true_val, false, &mut matches);
                    tally(*false_val, false, &mut matches);
                }
                ValueKind::ExtractLane { vector, index } => {
                    tally(*vector, false, &mut matches);
                    tally(*index, false, &mut matches);
                }
                ValueKind::InsertLane {
                    vector,
                    index,
                    value,
                } => {
                    tally(*vector, false, &mut matches);
                    tally(*index, false, &mut matches);
                    tally(*value, false, &mut matches);
                }
                ValueKind::ShuffleLanes { left, right, mask } => {
                    tally(*left, false, &mut matches);
                    tally(*right, false, &mut matches);
                    for &lane in self.value_pool.get(*mask) {
                        tally(lane, false, &mut matches);
                    }
                }
                ValueKind::ExtractField { aggregate, .. } => {
                    tally(*aggregate, false, &mut matches);
                }
                ValueKind::InsertField {
                    aggregate, value, ..
                } => {
                    tally(*aggregate, false, &mut matches);
                    tally(*value, false, &mut matches);
                }
                ValueKind::PtrOffset { base, indices, .. } => {
                    tally(*base, false, &mut matches);
                    for &index in self.value_pool.get(*indices) {
                        tally(index, false, &mut matches);
                    }
                }
                ValueKind::Load { addr, .. } | ValueKind::AtomicLoad { addr, .. } => {
                    tally(*addr, false, &mut matches);
                }
                ValueKind::Store { addr, value, .. }
                | ValueKind::AtomicStore { addr, value, .. }
                | ValueKind::AtomicReadModifyWrite { addr, value, .. } => {
                    tally(*addr, false, &mut matches);
                    tally(*value, false, &mut matches);
                }
                ValueKind::Alloca { count, .. } => {
                    tally(*count, false, &mut matches);
                }
                ValueKind::CompareAndSwap {
                    addr,
                    expected,
                    desired,
                    ..
                } => {
                    tally(*addr, false, &mut matches);
                    tally(*expected, false, &mut matches);
                    tally(*desired, false, &mut matches);
                }
                ValueKind::Freeze { value } | ValueKind::Match { input: value, .. } => {
                    tally(*value, false, &mut matches);
                }
                ValueKind::Intrinsic { args, .. } | ValueKind::Call { args, .. } => {
                    for &arg in self.value_pool.get(*args) {
                        tally(arg, false, &mut matches);
                    }
                }
                ValueKind::CallIndirect { callee, args, .. } => {
                    tally(*callee, false, &mut matches);
                    for &arg in self.value_pool.get(*args) {
                        tally(arg, false, &mut matches);
                    }
                }
                ValueKind::Project { call, .. } => {
                    tally(*call, false, &mut matches);
                }
                ValueKind::Const(_)
                | ValueKind::ConstPoolRef(_)
                | ValueKind::GlobalRef(_)
                | ValueKind::FuncAddr(_)
                | ValueKind::Fence { .. }
                | ValueKind::RegionParam { .. } => {}
            }
        }
        // Every region's results, function bodies included; this loop is
        // the single source for result tallies.
        for region_index in 0..self.regions.len() {
            for &result in self.region_results(RegionId(region_index as u32)) {
                tally(result, true, &mut matches);
            }
        }

        for (id, uses) in matches {
            if uses.other != 0 {
                errs.push(RVSDGVerificationError::PredicateNonConditionUse(id));
            }
            if uses.predicate_slots > 1 {
                errs.push(RVSDGVerificationError::PredicateUsedMoreThanOnce(
                    id,
                    uses.predicate_slots + uses.other,
                ));
            }
        }
    }
}
