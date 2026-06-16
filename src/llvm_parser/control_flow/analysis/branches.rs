//! Branch (section 4.2) continuation-point analysis and the small per-branch
//! helpers shared by the restructuring transform and the construction walk: the
//! continuation points of a fan-out, the switch control predicate, and join-phi
//! resolution.

use llvm_ir::{instruction::Phi, terminator::Switch};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx, block_mapper::BasicBlockId, control_flow::analysis::signature::phi_incoming_from,
        dominance::dominator_graph_of_arc, instructions::RegionLowerer,
    },
    rvsdg::{MatchArm, ValueId},
};

/// The **continuation points** of a fan-out (section 4.2): the tail blocks where
/// the region `entries` (e.g. a loop's exit vertices) reconverge, within the
/// acyclic region bounded by `boundary` (reaching a boundary block or the
/// synthetic exit stops the walk). Exactly one continuation means the entries
/// all rejoin at a single block (lowered as a plain join); more than one means
/// they reconverge at several places (lowered as a demux that routes each).
///
/// Classified by **region-local reachability**, not global dominance: each entry
/// is BFS-ed within the region, counting how many distinct entries reach each
/// block. A block reached from >=2 entries is a continuation point when it is
/// itself an entry, or has a predecessor reached by strictly fewer entries (the
/// point where those paths first join). An in-region edge `p -> v` means every
/// entry reaching `p` also reaches `v`, so `p`'s reach set is always a subset of
/// `v`'s -- hence "strictly fewer" is just a smaller reach count, no per-block set
/// needed. (Global dominance misclassifies cross-edges into an enclosing branch's
/// tail.) Returned in ascending block-id order for deterministic lowering.
#[tracing::instrument(skip_all)]
pub(in crate::llvm_parser) fn continuation_points(
    fn_ctx: &FnCtx,
    entries: &[BasicBlockId],
    boundary: &[BasicBlockId],
) -> SmallVec<[BasicBlockId; 4]> {
    let exit = fn_ctx.exit_block_id;
    let stops = |b: BasicBlockId| b == exit || boundary.contains(&b);

    // Per block: how many distinct entries reach it (`count`) and the stamp of the
    // entry whose BFS last visited it (`seen_by`). The stamp dedups repeated
    // visits within one entry's BFS so each entry counts that block at most once.
    let mut reach: FxHashMap<BasicBlockId, ReachCount> = FxHashMap::default();
    for (entry_index, &start) in entries.iter().enumerate() {
        if stops(start) {
            continue;
        }
        let stamp = entry_index as u32 + 1;
        let mut stack: SmallVec<[BasicBlockId; 16]> = SmallVec::new();
        stack.push(start);
        while let Some(block) = stack.pop() {
            let reached = reach.entry(block).or_default();
            if reached.seen_by == stamp {
                continue;
            }
            reached.seen_by = stamp;
            reached.count += 1;
            for &succ in fn_ctx.bb_mapper.outputs(block) {
                if !stops(succ) {
                    stack.push(succ);
                }
            }
        }
    }

    let mut points: SmallVec<[BasicBlockId; 4]> = SmallVec::new();
    for (&block, reached) in &reach {
        if reached.count < 2 {
            continue;
        }
        let joins_here = entries.contains(&block)
            || fn_ctx.bb_mapper.inputs(block).iter().any(|pred| {
                reach
                    .get(pred)
                    .is_some_and(|pred_reach| pred_reach.count < reached.count)
            });
        if joins_here {
            points.push(block);
        }
    }
    points.sort_unstable_by_key(|block| block.0);
    points
}

/// Per-block reach state used by [`continuation_points`]: how many distinct
/// entries reach the block, and the stamp of the entry whose BFS last touched it.
#[derive(Default)]
struct ReachCount {
    count: u32,
    seen_by: u32,
}

/// The **continuation points** of the branch at `head`: the blocks where its
/// arms (the `fan_out` targets) rejoin after diverging -- the tails the branch
/// reconverges into. Used to classify the branch (section 4.2): one continuation
/// is a plain split-join, several is a multi-way demux.
///
/// Each arm *owns* a subgraph -- the blocks reachable only through that one arm
/// and no other (`dominator_graph_of_arc`, restricted to the branch interior:
/// not a `boundary` block, not the synthetic exit). A continuation point is then
/// a block that an arm's subgraph leads out to but that belongs to *no* arm
/// exclusively -- the place two or more arms can meet. A fan-out target that is
/// also reachable from outside the branch (its owned subgraph is empty) is itself
/// such a meeting point.
///
/// Defining it by owned subgraphs rather than a plain reachability scan is what
/// correctly handles arms that exit to *different* enclosing boundary blocks:
/// each such block belongs to no arm, so each is its own continuation and the
/// branch is a demux -- a reachability scan would wrongly merge them.
#[tracing::instrument(skip_all)]
pub(in crate::llvm_parser) fn branch_continuation_points(
    fn_ctx: &FnCtx,
    head: BasicBlockId,
    fan_out: &[BasicBlockId],
    boundary: &[BasicBlockId],
) -> SmallVec<[BasicBlockId; 4]> {
    let exit = fn_ctx.exit_block_id;
    let in_region = |block: BasicBlockId| block != exit && !boundary.contains(&block);

    // One branch subgraph per fan-out arc (the paper's `Bj`): the dominator graph
    // of `head -> fan_out[j]`, confined to the branch interior.
    let branch_subgraphs: Vec<SmallVec<[BasicBlockId; 8]>> = fan_out
        .iter()
        .map(|&target| {
            dominator_graph_of_arc(
                fn_ctx.immediate_dominators,
                &fn_ctx.bb_mapper.blocks,
                in_region,
                (head, target),
            )
        })
        .collect();
    let mut in_any_subgraph: FxHashSet<BasicBlockId> = FxHashSet::default();
    for subgraph in &branch_subgraphs {
        in_any_subgraph.extend(subgraph.iter().copied());
    }

    let mut continuations: FxHashSet<BasicBlockId> = FxHashSet::default();
    // A successor of any branch subgraph that escapes every subgraph is a
    // continuation point (an in-region join, or an enclosing boundary the arm
    // exits to -- both are kept; only the synthetic exit is excluded).
    for subgraph in &branch_subgraphs {
        for &block in subgraph {
            for &succ in fn_ctx.bb_mapper.outputs(block) {
                if succ != exit && !in_any_subgraph.contains(&succ) {
                    continuations.insert(succ);
                }
            }
        }
    }
    // A fan-out target whose subgraph is empty (it is also reached from outside
    // the branch) is itself a continuation point.
    for &target in fan_out {
        if target != exit && !in_any_subgraph.contains(&target) {
            continuations.insert(target);
        }
    }

    let mut result: SmallVec<[BasicBlockId; 4]> = continuations.into_iter().collect();
    result.sort_unstable_by_key(|block| block.0);
    result
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Build the control/predicate value for a `switch` and its arm targets:
    /// arm 0 is the default, arms `1..=N` the cases in declaration order. The
    /// `match` maps each case value to its arm index; any other value to 0
    /// (default). Returns the control predicate and the arm-target list.
    pub(in crate::llvm_parser) fn switch_predicate(
        &mut self,
        switch: &Switch,
    ) -> color_eyre::Result<(ValueId, Vec<BasicBlockId>)> {
        let operand = self.operand(&switch.operand)?;
        let mut targets = Vec::with_capacity(switch.dests.len() + 1);
        targets.push(*self.fn_ctx.bb_mapper.get_expect(&switch.default_dest));
        let mut arms: Vec<MatchArm> = Vec::with_capacity(switch.dests.len());
        for (case_index, (case_const, dest)) in switch.dests.iter().enumerate() {
            targets.push(*self.fn_ctx.bb_mapper.get_expect(dest));
            // Case values are integer constants; read the value to key the match.
            let value = self
                .const_int_value(case_const)
                .ok_or_else(|| color_eyre::eyre::eyre!("switch case value is not an integer"))?;
            // Arm 0 is the default, so case `k` is arm `k + 1`.
            arms.push(MatchArm {
                value,
                alternative: case_index as u32 + 1,
            });
        }
        let alternatives = targets.len() as u32;
        let predicate = self.rb.match_op(operand, &arms, 0, alternatives);
        Ok((predicate, targets))
    }

    /// Resolve each join phi's contribution from *this* arm's scope: an inner gamma
    /// sharing the join may have already bound the phi destination (use it);
    /// otherwise take the phi's incoming for `exit_pred` (the predecessor the arm
    /// walk exited through) and resolve it here.
    pub(in crate::llvm_parser) fn resolve_arm_join_phis(
        &mut self,
        phis_at_join: &[&Phi],
        exit_pred: Option<BasicBlockId>,
    ) -> color_eyre::Result<Vec<ValueId>> {
        phis_at_join
            .iter()
            .map(|phi| {
                if let Some(&value) = self.name_to_value.get(&phi.dest) {
                    return Ok(value);
                }
                let incoming = exit_pred.and_then(|pred| {
                    phi_incoming_from(phi, self.fn_ctx.bb_mapper, |block| block == pred)
                });
                if let Some((operand, _)) = incoming {
                    return self.operand(operand);
                }
                Err(color_eyre::eyre::eyre!(
                    "phi {:?} at join has no incoming value from this arm (exit_pred {:?})",
                    phi.dest,
                    exit_pred,
                ))
            })
            .collect()
    }

    /// Read an integer `switch` case value from its constant operand.
    fn const_int_value(&self, constant: &llvm_ir::ConstantRef) -> Option<i64> {
        match constant.as_ref() {
            llvm_ir::Constant::Int { value, .. } => Some(*value as i64),
            _ => None,
        }
    }
}
