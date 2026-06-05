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

#[cfg(test)]
pub(super) mod test_fixture;

use llvm_ir::{Instruction, terminator::CondBr};

use crate::{
    llvm_parser::{
        block_mapper::BasicBlockId, instructions::RegionLowerer, scc_tree::SccTreeNodeId,
    },
    rvsdg::{State, ValueId},
};

/// What a `lower_region` call produced at its exit point.
///
/// - `AtBoundary` is returned when the region exits at its `end` block (a
///   gamma-arm join) or at the synthetic function exit. There are no result
///   values to wire out: a gamma arm computes its result values from phi
///   contributions in the join block, not from a terminator.
/// - `Returned` is returned when the region terminated via `Ret`, carrying
///   the function's return operand (empty for void returns).
#[derive(Debug)]
pub enum RegionExit {
    AtBoundary(State),
    Returned { state: State, values: Vec<ValueId> },
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
        end: Option<BasicBlockId>,
        skip_loop_dispatch_at: Option<BasicBlockId>,
    ) -> color_eyre::Result<RegionExit> {
        let mut current = start;
        let mut state = entry_state;

        loop {
            // Hit the region's caller-supplied boundary (typically a
            // gamma-arm join block).
            if end == Some(current) {
                return Ok(RegionExit::AtBoundary(state));
            }

            // Hit the synthetic function-exit block (added by the parser to
            // give every `Ret` a common destination).
            if current == self.fn_ctx.exit_block_id {
                return Ok(RegionExit::AtBoundary(state));
            }

            // A block that is the entry vertex of a strongly connected
            // component starts a loop; lower the whole component as one
            // theta node and resume at the component's single exit. The
            // caller-supplied skip filters out the one entry vertex
            // belonging to the loop we are already inside (see doc
            // comment).
            if Some(current) != skip_loop_dispatch_at {
                if let Some(scc_id) = self.loop_at(current) {
                    let (next_state, exit_target) = self.lower_scc_as_theta(state, scc_id)?;
                    state = next_state;
                    current = exit_target;
                    continue;
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
                    current = *next;
                }
                llvm_ir::Terminator::CondBr(cond_br) => {
                    // A two-way conditional branch lowers to a binary
                    // gamma node. The gamma's join point is the immediate
                    // post-dominator of the branching block: the unique
                    // place every path leaving either arm must converge
                    // before continuing in the surrounding region.
                    //
                    // We require that an immediate post-dominator exists
                    // for every conditional branch we encounter. This is a
                    // structured-branches assumption: both arms must
                    // reconverge inside this region before the region
                    // ends. Bahmann, Reissmann, Jahre, Meyer (2015)
                    // section 4.2 handles the unstructured case by
                    // partitioning the acyclic subgraph after a fan-out
                    // into a head, several branch subgraphs, and a tail,
                    // then introducing an auxiliary continuation
                    // predicate p with assignments inside each branch
                    // subgraph and a dispatching branch on p inside the
                    // tail. That transform has not been implemented yet.
                    //
                    // Until it lands, sources with unstructured branches
                    // (early returns inside conditionals, breaks out of
                    // nested gammas, switch statements with shared
                    // fall-through tails) panic here.
                    let join = self.fn_ctx.post_immediate_dominators[current.0 as usize]
                        .expect(
                            "conditional branch has no immediate post-dominator: \
                             arms do not reconverge inside the surrounding region. \
                             Needs the not-yet-implemented continuation-predicate \
                             transform (Bahmann et al. 2015 section 4.2).",
                        );
                    state = self.lower_cond_branch(state, cond_br, join)?;
                    current = join;
                }
                llvm_ir::Terminator::Switch(switch) => {
                    // An n-way switch lowers to an n-arm gamma node. Same
                    // post-dominator requirement and same continuation-
                    // predicate gap as the conditional branch case above.
                    let join = self.fn_ctx.post_immediate_dominators[current.0 as usize]
                        .expect(
                            "switch terminator has no immediate post-dominator: \
                             arms do not reconverge inside the surrounding region. \
                             Needs the not-yet-implemented continuation-predicate \
                             transform (Bahmann et al. 2015 section 4.2).",
                        );
                    state = self.lower_switch(state, switch, join)?;
                    current = join;
                }
                llvm_ir::Terminator::Unreachable(_) => {
                    return Ok(RegionExit::AtBoundary(state));
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
}
