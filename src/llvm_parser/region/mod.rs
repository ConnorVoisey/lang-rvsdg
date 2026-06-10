//! Lower a single function's CFG into an RVSDG region.
//!
//! The construction walks each basic block in source order, threading a
//! state value through every instruction and dispatching to specialised
//! lowering paths at the control-flow joins:
//!
//! - A block that is the entry vertex of a strongly connected component
//!   starts a loop and dispatches into `loops::lower_scc_as_theta`,
//!   which emits a theta node and resumes lowering at the loop's
//!   single exit target.
//! - A conditional branch (`CondBr` or `Switch`) dispatches into
//!   `branches::lower_cond_branch` / `branches::lower_switch`, which
//!   emit a gamma node and resume lowering at the post-dominator join.
//!
//! Both lowering paths add their own `impl<'rb, 'g, 'm> RegionLowerer`
//! blocks contributing to the single `RegionLowerer` struct defined in
//! `instructions.rs`. Rust permits multiple `impl` blocks for the same
//! type across modules; the type stays one definition, and the methods
//! are sorted by topic.

pub mod branches;
pub mod loops;
pub mod phi;
pub mod restructure;

#[cfg(test)]
pub(super) mod test_fixture;

use llvm_ir::{Instruction, terminator::CondBr};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        block_mapper::BasicBlockId,
        instructions::RegionLowerer,
        region::phi::{phi_incoming_from, phi_instructions_at},
        scc_tree::SccTreeNodeId,
    },
    rvsdg::{State, ValueId},
};

/// What a `lower_region` call produced at its exit point.
///
/// - `AtBoundary` is returned when the region exits at its `end` block (a
///   gamma-arm join) or at the synthetic function exit. `exit_pred` is the
///   block the walk reached the boundary *from* (the predecessor of `end`
///   along this path), or `None` when the boundary was reached without a
///   meaningful predecessor (the start block already being the boundary
///   gives the seeded entry predecessor). A gamma arm uses `exit_pred` to
///   pick its contribution to each join phi when no inner gamma already
///   bound it.
/// - `Returned` is returned when the region terminated via `Ret`, carrying
///   the function's return operand (empty for void returns).
#[derive(Debug)]
pub enum RegionExit {
    AtBoundary {
        state: State,
        exit_pred: Option<BasicBlockId>,
        /// Which boundary block the walk stopped at (the synthetic function
        /// exit, or one of the caller's boundary blocks). `None` for a dead
        /// end (`Unreachable`). With a multi-block boundary this is how the
        /// caller learns *which* continuation an arm reached — the paper's
        /// `p` value.
        reached: Option<BasicBlockId>,
    },
    Returned {
        state: State,
        values: Vec<ValueId>,
    },
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Lower a region from `start` to `end` (exclusive) or to a function
    /// exit (`Ret` / `Unreachable` / the synthetic exit block), threading
    /// state through each instruction and each nested gamma/theta node.
    ///
    /// The returned `RegionExit` tells the caller which kind of exit
    /// happened. The caller wires this into the enclosing region: a
    /// gamma-arm `BranchResult`, a theta result, or a function `FnResult`.
    ///
    /// `skip_loop_dispatch_at`, when `Some(block)`, suppresses the
    /// loop-header dispatch at that specific block. This is how
    /// `lower_scc_as_theta` walks its own loop body without infinitely
    /// recursing: it passes `Some(header)` so that when this method
    /// reaches the loop header it treats it as an ordinary block (lower
    /// its instructions, follow its terminator) rather than dispatching
    /// back into `lower_scc_as_theta`. Inner loops encountered later in
    /// the walk are unaffected and dispatch normally.
    pub fn lower_region(
        &mut self,
        entry_state: State,
        start: BasicBlockId,
        boundary: &[BasicBlockId],
        skip_loop_dispatch_at: Option<BasicBlockId>,
        entry_prev: Option<BasicBlockId>,
    ) -> color_eyre::Result<RegionExit> {
        let mut current = start;
        let mut state = entry_state;
        // The block we arrived at `current` from. Seeded with `entry_prev`
        // (the branching head for a gamma arm) so an interior-join phi at
        // the arm's first block resolves correctly. Set to `None` after a
        // gamma/theta dispatch, whose join/exit phis are already bound by
        // that dispatch and must not be re-resolved here.
        let mut prev = entry_prev;

        loop {
            // Hit one of the region's caller-supplied boundary blocks
            // (typically gamma-arm joins / continuation points). Report the
            // block we reached it from (so an arm can pick its join-phi
            // contribution) and which boundary block it was (the `p` the
            // caller demultiplexes on).
            if boundary.contains(&current) {
                return Ok(RegionExit::AtBoundary {
                    state,
                    exit_pred: prev,
                    reached: Some(current),
                });
            }

            // Hit the synthetic function-exit block (added by the parser to
            // give every `Ret` a common destination).
            if current == self.fn_ctx.exit_block_id {
                return Ok(RegionExit::AtBoundary {
                    state,
                    exit_pred: prev,
                    reached: Some(current),
                });
            }

            // A multi-entry (irreducible) SCC is lowered at its dispatch
            // dominator: the `q` entry-predicate is computed in the branch
            // structure from here to the loop's entries, then a single
            // theta runs the loop. Done before the per-entry loop dispatch
            // below so control never reaches an individual entry vertex.
            if Some(current) != skip_loop_dispatch_at {
                if let Some(scc_id) = self.multi_entry_dispatch_at(current) {
                    let (next_state, exit_target) =
                        self.lower_multi_entry_dispatch(state, scc_id, current)?;
                    state = next_state;
                    current = exit_target;
                    prev = None;
                    continue;
                }
            }

            // A block that is the entry vertex of a strongly connected
            // component starts a loop; lower the whole component as one
            // theta node and resume at the component's single exit. The
            // caller-supplied skip filters out the one entry vertex
            // belonging to the loop we are already inside (see doc
            // comment).
            if Some(current) != skip_loop_dispatch_at {
                if let Some(scc_id) = self.loop_at(current) {
                    let (next_state, exit_target) =
                        self.lower_scc_as_theta(state, scc_id, current)?;
                    state = next_state;
                    current = exit_target;
                    // The loop bound any loop-closed phis at its exit target;
                    // don't re-resolve them from a stale predecessor.
                    prev = None;
                    continue;
                }
            }

            // Interior-join phi resolution. When `current` is reached
            // linearly (via a `Br`, or as a gamma arm's first block) from a
            // known predecessor, bind any phi destinations here to the
            // incoming value for that predecessor. This is what lets two
            // arms that fall through to (or otherwise share) the same tail
            // block each lower it path-aware — the shared block's phis pick
            // the value for the path that arrived. Reached-via-gamma/theta
            // joins have `prev = None` and are skipped (already bound).
            if let Some(pred) = prev {
                let bb = &self.fn_ctx.func.basic_blocks[current.0 as usize];
                let phis = phi_instructions_at(bb);
                for phi in &phis {
                    if let Some((op, _)) =
                        phi_incoming_from(phi, self.fn_ctx.bb_mapper, |id| id == pred)
                    {
                        let value = self.operand(op)?;
                        self.name_to_value.insert(phi.dest.clone(), value);
                    }
                }
            }

            // Straight-line block: lower its non-phi instructions, then
            // dispatch on the terminator. Phis are absorbed into gamma-arm
            // result wiring elsewhere (see `arm_phi_contributions`).
            state = self.lower_instructions_skip_phis(state, current)?;

            // Re-borrow `bb` after the mutating call so we can read its
            // terminator. The previous mutable borrow ended above.
            let bb = &self.fn_ctx.func.basic_blocks[current.0 as usize];
            match &bb.term {
                llvm_ir::Terminator::Ret(ret) => {
                    let values = match &ret.return_operand {
                        Some(op) => vec![self.operand(op)?],
                        None => Vec::new(),
                    };
                    return Ok(RegionExit::Returned { state, values });
                }
                llvm_ir::Terminator::Br(br) => {
                    let next = self.fn_ctx.bb_mapper.get_expect(&br.dest);
                    prev = Some(current);
                    current = *next;
                }
                llvm_ir::Terminator::CondBr(cond_br) => {
                    // A two-way conditional branch lowers to a binary
                    // gamma node. The gamma's join point is the immediate
                    // post-dominator of the branching block: the unique
                    // place every path leaving either arm must converge
                    // before continuing in the surrounding region.
                    //
                    // The arms are walked path-aware to this join (see the
                    // interior-join phi resolution above and
                    // `arm_phi_contributions`), so shared continuations —
                    // switch fall-through tails, cross edges where two arms
                    // reach the same block — are handled even though they
                    // are dominated by no single arm.
                    //
                    // The remaining unhandled shape is a branch with NO
                    // common post-dominator at all (arms that never
                    // reconverge before the function exit). Bahmann et al.
                    // 2015 §4.2's auxiliary continuation predicate `p`
                    // covers it; with this compiler's synthetic unified
                    // exit block every branch has at least that as a
                    // post-dominator, so the `.expect` is a guard against a
                    // genuinely malformed/unreachable CFG rather than the
                    // common path.
                    let true_target = *self.fn_ctx.bb_mapper.get_expect(&cond_br.true_dest);
                    let false_target = *self.fn_ctx.bb_mapper.get_expect(&cond_br.false_dest);
                    let arm_targets = [true_target, false_target];
                    let continuation_points =
                        restructure::continuation_points(self.fn_ctx, &arm_targets, boundary);
                    let join = self
                        .resolve_branch_join(current, &continuation_points)
                        .ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "conditional branch at block {} has no post-dominator: \
                                 its arms never reconverge, not even at the function exit",
                                current.0
                            )
                        })?;
                    if continuation_points.len() > 1
                        && self.arms_reconverge(&arm_targets, &continuation_points, join)
                    {
                        // §4.2 multi-continuation: p-demux, lowered once.
                        let predicate = self.operand(&cond_br.condition)?;
                        state = self.lower_multi_continuation_branch(
                            state,
                            predicate,
                            &arm_targets,
                            &continuation_points,
                            current,
                            join,
                        )?;
                    } else {
                        state = self.lower_cond_branch(state, cond_br, current, join)?;
                    }
                    // The gamma(s) bound `join`'s phis; resume there with no
                    // linear predecessor so they are not re-resolved.
                    prev = None;
                    current = join;
                }
                llvm_ir::Terminator::Switch(switch) => {
                    // An n-way switch lowers to an n-arm gamma node. Same
                    // post-dominator requirement and same continuation-
                    // predicate gap as the conditional branch case above.
                    let mut arm_targets: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
                    arm_targets.push(*self.fn_ctx.bb_mapper.get_expect(&switch.default_dest));
                    for (_, dest) in &switch.dests {
                        arm_targets.push(*self.fn_ctx.bb_mapper.get_expect(dest));
                    }
                    let continuation_points =
                        restructure::continuation_points(self.fn_ctx, &arm_targets, boundary);
                    let join = self
                        .resolve_branch_join(current, &continuation_points)
                        .ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "switch at block {} has no post-dominator: \
                                 its arms do not reconverge inside the surrounding region",
                                current.0
                            )
                        })?;
                    if continuation_points.len() > 1
                        && self.arms_reconverge(&arm_targets, &continuation_points, join)
                    {
                        let (selector, targets) = self.switch_selector(switch)?;
                        state = self.lower_multi_continuation_branch(
                            state,
                            selector,
                            &targets,
                            &continuation_points,
                            current,
                            join,
                        )?;
                    } else {
                        state = self.lower_switch(state, switch, current, join)?;
                    }
                    prev = None;
                    current = join;
                }
                llvm_ir::Terminator::Unreachable(_) => {
                    return Ok(RegionExit::AtBoundary {
                        state,
                        exit_pred: prev,
                        reached: None,
                    });
                }
                t => todo!("handle terminator: {t:?}"),
            }
        }
    }

    /// Lower every non-phi instruction in `block` in source order, threading
    /// state through each. Does NOT touch the terminator and does NOT walk
    /// any successor block: pure per-block work.
    ///
    /// Phi instructions are skipped here because they're handled separately
    /// at region boundaries: header phis are seeded as theta params, gamma
    /// join phis are bound by the post-gamma wiring in `lower_n_way_branch`,
    /// and loop-closed phis at a loop's exit_target are bound by
    /// `lower_scc_as_theta`.
    ///
    /// Used by `lower_region` for its per-block walk and by the do-while
    /// theta body lowering to lower its cond_block manually after the
    /// body region.
    pub(super) fn lower_instructions_skip_phis(
        &mut self,
        mut state: State,
        block: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let bb = &self.fn_ctx.func.basic_blocks[block.0 as usize];
        for inst in &bb.instrs {
            if matches!(inst, Instruction::Phi(_)) {
                continue;
            }
            state = self.lower_instruction(state, inst)?;
        }
        Ok(state)
    }

    /// Lower the non-phi non-terminator instructions of `cond_block` in
    /// the current region, then resolve the conditional branch's
    /// condition operand to a `ValueId` in this region's scope. Returns
    /// the state after the lowering and the resolved condition value.
    ///
    /// Used by the test-first lowering path in `lower_scc_as_theta` to
    /// evaluate the loop condition at the start of each theta body
    /// iteration, before the gating gamma decides whether to run the
    /// body work or pass loop variables through unchanged.
    ///
    /// The cond_block's instruction destinations are bound in this
    /// region's `name_to_value`. They are local to this region (the
    /// gating gamma's arm regions have their own `name_to_value` and
    /// do not inherit), so no cleanup is needed.
    pub(super) fn lower_cond_block_in_region(
        &mut self,
        state: State,
        cond_block: BasicBlockId,
        cond_br: &CondBr,
    ) -> color_eyre::Result<(State, ValueId)> {
        let state = self.lower_instructions_skip_phis(state, cond_block)?;
        let cond_value = self.operand(&cond_br.condition)?;
        Ok((state, cond_value))
    }

    /// If `id` is an entry vertex of a strongly connected component in
    /// the SCC tree, return that component's identifier. Returns `None`
    /// for blocks that are not entry vertices (including blocks inside a
    /// component's body but not at its entry).
    fn loop_at(&self, id: BasicBlockId) -> Option<SccTreeNodeId> {
        self.fn_ctx.scc_entry_block_to_id[id.0 as usize]
    }

    /// The block at which the build resumes after the branch terminating
    /// `head`. With exactly one continuation point that IS the join. With
    /// more than one (the `p`-demux) it is the post-dominator — the single
    /// point all the continuations reconverge at, where the demux's outputs
    /// land. Zero continuation points (arms that all return / exit) also use
    /// the post-dominator, which the synthetic exit makes the function exit.
    fn resolve_branch_join(
        &self,
        head: BasicBlockId,
        continuation_points: &[BasicBlockId],
    ) -> Option<BasicBlockId> {
        if continuation_points.len() == 1 {
            Some(continuation_points[0])
        } else {
            self.fn_ctx.post_immediate_dominators[head.0 as usize]
        }
    }

    /// Whether the `p`-demux can lower this branch: every arm, walked to the
    /// demux-target boundary (the continuation points plus `join`), must
    /// actually reach one of those targets. An arm that instead escapes the
    /// region — returning (reaching the synthetic exit) or hitting
    /// `unreachable` first — cannot be expressed as a demux output yet, so
    /// the caller falls back to the single-join path (which clones the
    /// shared continuation but handles the escaping arm). Fixture 34's
    /// done/spin branch is such a case.
    fn arms_reconverge(
        &self,
        arm_targets: &[BasicBlockId],
        continuation_points: &[BasicBlockId],
        join: BasicBlockId,
    ) -> bool {
        let exit = self.fn_ctx.exit_block_id;
        // A join at the function exit means the branch never reconverges at a
        // real block (its arms return / diverge); the demux can't express
        // that, so let the single-join path handle it.
        if join == exit {
            return false;
        }
        let is_target = |b: BasicBlockId| b == join || continuation_points.contains(&b);

        // Forward region: every block an arm can reach before a demux target.
        // Sinks (the function exit, `unreachable` blocks, infinite sub-loops)
        // are included so we can detect them below. Bounded to the region —
        // the walk never crosses a target, so it cannot run away through the
        // rest of the function.
        let mut region: FxHashSet<BasicBlockId> = FxHashSet::default();
        let mut stack: SmallVec<[BasicBlockId; 16]> = SmallVec::new();
        for &arm in arm_targets {
            if !is_target(arm) {
                stack.push(arm);
            }
        }
        while let Some(b) = stack.pop() {
            if !region.insert(b) {
                continue;
            }
            for &succ in self.fn_ctx.bb_mapper.outputs(b) {
                if !is_target(succ) && !region.contains(&succ) {
                    stack.push(succ);
                }
            }
        }

        // Reverse reachability of a demux target, restricted to the region
        // (only region predecessors are followed, so this is also bounded).
        let mut can_reach: FxHashSet<BasicBlockId> = FxHashSet::default();
        let mut reverse: SmallVec<[BasicBlockId; 16]> = SmallVec::new();
        for &t in continuation_points {
            reverse.push(t);
        }
        reverse.push(join);
        while let Some(b) = reverse.pop() {
            for &pred in self.fn_ctx.bb_mapper.inputs(b) {
                if region.contains(&pred) && can_reach.insert(pred) {
                    reverse.push(pred);
                }
            }
        }

        // The demux can lower this branch iff every block an arm reaches can
        // itself reach a target — i.e. the region has no sink. A region block
        // that cannot reach a target is a return / trap / infinite loop, which
        // the demux cannot express; fall back to the single-join path.
        region.iter().all(|b| can_reach.contains(b))
    }
}
