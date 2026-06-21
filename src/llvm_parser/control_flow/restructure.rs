//! **Phase 1 of the control-flow pipeline: the restructuring transform**
//! (Bahmann, Reissmann, Jahre, Meyer 2015, section 4). Materializes a Structured
//! Region Tree ([`super::rst`]) from the LLVM control flow graph plus the
//! per-function analyses in [`FnCtx`]. Phase 2 ([`super::construct`]) then walks
//! the RST and emits the RVSDG.
//!
//! A region's *role* (where it sits in an enclosing construct) fixes which exits
//! are valid: a sequential region may return, a loop body may produce a mixed
//! demux, a capture arm may only route, and so on. Each role has its own exit
//! enum so the construction walk handles exactly the shapes its role permits --
//! invalid combinations are unrepresentable rather than runtime errors. The
//! shared item walk ([`walk_items`]) is role-agnostic; the per-role producers
//! ([`structure_seq`] and friends) interpret its terminal into a role exit.

use color_eyre::eyre::eyre;
use smallvec::SmallVec;

use crate::llvm_parser::{
    FnCtx,
    block_mapper::BasicBlockId,
    control_flow::{
        analysis::branches::{branch_continuation_points, continuation_points},
        rst::{
            CaptureExit, CaptureRegion, DemuxBranchTarget, DemuxSpec, DemuxTail, EntryExit,
            EntryRegion, ExitDemux, ExitMerge, GammaMerge, GammaNode, LoopBodyExit, LoopBodyRegion,
            LoopCaptureExit, LoopCaptureRegion, RegionItem, SeqExit, SeqRegion, ThetaKind,
            ThetaNode,
        },
    },
};

/// The terminal of a region's item walk, before it is interpreted into a
/// role-specific exit. [`walk_items`] classifies the branch shape; each role
/// producer turns the terminal into the exit its role permits, erroring on the
/// shapes its role forbids.
enum RawTail {
    /// A boundary block (or the synthetic exit) was reached.
    Continuation {
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
    },
    /// A `ret` terminator.
    Return { block: BasicBlockId },
    /// An `unreachable` (or non-returning tail).
    Diverge,
    /// A branch every continuation of which is an enclosing boundary: a router.
    Route {
        head: BasicBlockId,
        arm_targets: SmallVec<[BasicBlockId; 4]>,
    },
    /// A branch all arms of which return/diverge (post-dominator is the synthetic
    /// exit): an all-arms-return gamma.
    ReturnGamma {
        head: BasicBlockId,
        arm_targets: SmallVec<[BasicBlockId; 4]>,
    },
    /// A branch whose continuation points mix in-region points and enclosing
    /// boundaries.
    MixedDemux {
        head: BasicBlockId,
        arm_targets: SmallVec<[BasicBlockId; 4]>,
        continuations: SmallVec<[BasicBlockId; 4]>,
    },
    /// A multi-exit loop whose exit tails do not reconverge (every one returns or
    /// diverges): the loop is the region's terminal and the post-theta dispatch
    /// returns the merged value.
    LoopReturn { theta: ThetaNode },
}

/// What [`structure_exit_demux`] decides a loop's exit vertices do after the
/// theta. The two reconverging shapes keep the theta a mid-region item; the
/// terminal shape makes the loop the enclosing region's terminal.
enum ExitOutcome {
    /// A single exit vertex: no demux, the enclosing region resumes here.
    Resume { target: BasicBlockId },
    /// Multiple exits reconverging at `join`: the enclosing region resumes there.
    Reconverge {
        demux: ExitDemux,
        join: BasicBlockId,
    },
    /// Multiple exits, every tail returning or diverging: a terminal return demux.
    Return { demux: ExitDemux },
}

/// Restructure `func` (described by `fn_ctx`) into its Structured Region Tree.
pub(in crate::llvm_parser) fn restructure_fn(fn_ctx: &FnCtx) -> color_eyre::Result<SeqRegion> {
    structure_seq(fn_ctx, BasicBlockId(0), &[], None, None)
}

/// The successor blocks of `head`'s branch terminator, in alternative order:
/// `CondBr` -> `[true, false]`; `Switch` -> `[default, case0, case1, ...]`.
pub(in crate::llvm_parser) fn arm_target_blocks(
    fn_ctx: &FnCtx,
    head: BasicBlockId,
) -> color_eyre::Result<SmallVec<[BasicBlockId; 4]>> {
    let bb = &fn_ctx.func.basic_blocks[head.0 as usize];
    let mut targets: SmallVec<[BasicBlockId; 4]> = SmallVec::new();
    match &bb.term {
        llvm_ir::Terminator::CondBr(cond_br) => {
            targets.push(*fn_ctx.bb_mapper.get_expect(&cond_br.true_dest));
            targets.push(*fn_ctx.bb_mapper.get_expect(&cond_br.false_dest));
        }
        llvm_ir::Terminator::Switch(switch) => {
            targets.push(*fn_ctx.bb_mapper.get_expect(&switch.default_dest));
            for (_, dest) in &switch.dests {
                targets.push(*fn_ctx.bb_mapper.get_expect(dest));
            }
        }
        other => {
            return Err(eyre!(
                "arm_target_blocks at block {} whose terminator is {:?}",
                head.0,
                other
            ));
        }
    }
    Ok(targets)
}

/// Structure a [`SeqRegion`] (top-level body, gamma split arm, or demux tail):
/// the items, then a [`SeqExit`].
fn structure_seq(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
    entry_prev: Option<BasicBlockId>,
    loop_header: Option<BasicBlockId>,
) -> color_eyre::Result<SeqRegion> {
    let (items, tail) = walk_items(fn_ctx, start, boundary, entry_prev, loop_header)?;
    let exit = match tail {
        RawTail::Continuation { reached, via } => SeqExit::ToContinuation { reached, via },
        RawTail::Return { block } => SeqExit::Return { block },
        RawTail::Diverge => SeqExit::Diverge,
        RawTail::ReturnGamma { head, arm_targets } => {
            let mut arms = Vec::with_capacity(arm_targets.len());
            for &target in &arm_targets {
                arms.push(structure_seq(fn_ctx, target, boundary, Some(head), None)?);
            }
            SeqExit::ReturnGamma { head, arms }
        }
        RawTail::LoopReturn { theta } => SeqExit::LoopReturn { theta },
        RawTail::Route { head, .. } => {
            return Err(eyre!(
                "router region (every continuation a boundary) at block {} in a sequential context",
                head.0
            ));
        }
        RawTail::MixedDemux { head, .. } => {
            return Err(eyre!(
                "mixed in-region/boundary demux at block {} outside a loop body",
                head.0
            ));
        }
    };
    Ok(SeqRegion { items, exit })
}

/// Structure a [`CaptureRegion`] (continuation-demux head arm): reach a demux
/// target, diverge, or route on.
fn structure_capture(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
    entry_prev: Option<BasicBlockId>,
) -> color_eyre::Result<CaptureRegion> {
    let (items, tail) = walk_items(fn_ctx, start, boundary, entry_prev, None)?;
    let exit = match tail {
        RawTail::Continuation { reached, via } => CaptureExit::ToContinuation { reached, via },
        RawTail::Diverge => CaptureExit::Diverge,
        RawTail::Route { head, arm_targets } => {
            let mut arms = Vec::with_capacity(arm_targets.len());
            for &target in &arm_targets {
                arms.push(structure_capture(fn_ctx, target, boundary, Some(head))?);
            }
            CaptureExit::Route { head, arms }
        }
        RawTail::Return { block } => {
            return Err(eyre!(
                "early return at block {} inside a continuation-demux arm",
                block.0
            ));
        }
        RawTail::ReturnGamma { head, .. } => {
            return Err(eyre!(
                "all-arms-return branch at block {} inside a continuation-demux arm",
                head.0
            ));
        }
        RawTail::MixedDemux { head, .. } => {
            return Err(eyre!(
                "nested mixed demux at block {} inside a continuation-demux arm is not supported",
                head.0
            ));
        }
        RawTail::LoopReturn { theta } => {
            return Err(eyre!(
                "non-reconverging multi-exit loop (exits {:?}) inside a continuation-demux arm is \
                 not handled",
                theta.exit_blocks
            ));
        }
    };
    Ok(CaptureRegion { items, exit })
}

/// Structure an [`EntryRegion`] (irreducible-loop entry region): reach an entry
/// vertex or route on.
fn structure_entry(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
    loop_header: Option<BasicBlockId>,
) -> color_eyre::Result<EntryRegion> {
    let (items, tail) = walk_items(fn_ctx, start, boundary, None, loop_header)?;
    let exit = match tail {
        RawTail::Continuation { reached, via } => EntryExit::ToContinuation { reached, via },
        RawTail::Route { head, arm_targets } => {
            let mut arms = Vec::with_capacity(arm_targets.len());
            for &target in &arm_targets {
                arms.push(structure_entry(fn_ctx, target, boundary, None)?);
            }
            EntryExit::Route { head, arms }
        }
        RawTail::Return { block } => {
            return Err(eyre!(
                "early return at block {} inside an irreducible-loop entry region",
                block.0
            ));
        }
        RawTail::ReturnGamma { head, .. } => {
            return Err(eyre!(
                "all-arms-return branch at block {} inside an irreducible-loop entry region",
                head.0
            ));
        }
        RawTail::Diverge => {
            return Err(eyre!(
                "divergence inside an irreducible-loop entry region is not handled"
            ));
        }
        RawTail::MixedDemux { head, .. } => {
            return Err(eyre!(
                "mixed demux at block {} inside an irreducible-loop entry region is not handled",
                head.0
            ));
        }
        RawTail::LoopReturn { theta } => {
            return Err(eyre!(
                "non-reconverging multi-exit loop (exits {:?}) inside an irreducible-loop entry \
                 region is not handled",
                theta.exit_blocks
            ));
        }
    };
    Ok(EntryRegion { items, exit })
}

/// Structure a [`LoopBodyRegion`]: reach a loop boundary, route over loop
/// boundaries, or end in a mixed in-body/boundary demux.
fn structure_loop_body(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
    entry_prev: Option<BasicBlockId>,
    loop_header: Option<BasicBlockId>,
) -> color_eyre::Result<LoopBodyRegion> {
    let (items, tail) = walk_items(fn_ctx, start, boundary, entry_prev, loop_header)?;
    let exit = match tail {
        RawTail::Continuation { reached, via } => LoopBodyExit::ToContinuation { reached, via },
        RawTail::Route { head, arm_targets } => {
            let mut arms = Vec::with_capacity(arm_targets.len());
            for &target in &arm_targets {
                arms.push(structure_loop_body(
                    fn_ctx,
                    target,
                    boundary,
                    Some(head),
                    None,
                )?);
            }
            LoopBodyExit::Route { head, arms }
        }
        RawTail::MixedDemux {
            head,
            arm_targets,
            continuations,
        } => {
            let in_region =
                |block: BasicBlockId| block != fn_ctx.exit_block_id && !boundary.contains(&block);

            // Head arms walk to whichever continuation they reach (capture arms).
            let mut head_boundary: SmallVec<[BasicBlockId; 8]> =
                continuations.iter().copied().collect();
            head_boundary.extend_from_slice(boundary);
            let mut arms = Vec::with_capacity(arm_targets.len());
            for &target in &arm_targets {
                arms.push(structure_loop_capture(
                    fn_ctx,
                    target,
                    &head_boundary,
                    Some(head),
                )?);
            }

            // Each in-region continuation gets a tail lowered to the loop boundary.
            let mut targets: Vec<DemuxBranchTarget> = Vec::with_capacity(continuations.len());
            for &continuation in &continuations {
                let in_region_tail = if in_region(continuation) {
                    Some(structure_loop_body(
                        fn_ctx,
                        continuation,
                        boundary,
                        None,
                        None,
                    )?)
                } else {
                    None
                };
                targets.push(DemuxBranchTarget {
                    block: continuation,
                    in_region_tail,
                });
            }
            LoopBodyExit::Demux {
                head,
                arms,
                targets,
            }
        }
        RawTail::Return { block } => {
            return Err(eyre!(
                "early return at block {} inside a loop body is not handled",
                block.0
            ));
        }
        RawTail::ReturnGamma { head, .. } => {
            return Err(eyre!(
                "all-arms-return branch at block {} inside a loop body is not handled",
                head.0
            ));
        }
        RawTail::Diverge => {
            return Err(eyre!("divergence inside a loop body is not handled"));
        }
        RawTail::LoopReturn { theta } => {
            return Err(eyre!(
                "non-reconverging multi-exit loop (exits {:?}) inside a loop body is not handled",
                theta.exit_blocks
            ));
        }
    };
    Ok(LoopBodyRegion { items, exit })
}

/// Structure a [`LoopCaptureRegion`] (loop-body-demux head arm): reach a
/// continuation or route on.
fn structure_loop_capture(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
    entry_prev: Option<BasicBlockId>,
) -> color_eyre::Result<LoopCaptureRegion> {
    let (items, tail) = walk_items(fn_ctx, start, boundary, entry_prev, None)?;
    let exit = match tail {
        RawTail::Continuation { reached, via } => LoopCaptureExit::ToContinuation { reached, via },
        RawTail::Route { head, arm_targets } => {
            let mut arms = Vec::with_capacity(arm_targets.len());
            for &target in &arm_targets {
                arms.push(structure_loop_capture(
                    fn_ctx,
                    target,
                    boundary,
                    Some(head),
                )?);
            }
            LoopCaptureExit::Route { head, arms }
        }
        RawTail::Return { block } => {
            return Err(eyre!(
                "early return at block {} inside a loop-demux arm is not handled",
                block.0
            ));
        }
        RawTail::ReturnGamma { head, .. } => {
            return Err(eyre!(
                "all-arms-return branch at block {} inside a loop-demux arm is not handled",
                head.0
            ));
        }
        RawTail::Diverge => {
            return Err(eyre!("divergence inside a loop-demux arm is not handled"));
        }
        RawTail::MixedDemux { head, .. } => {
            return Err(eyre!(
                "nested mixed demux at block {} inside a loop-demux arm is not supported",
                head.0
            ));
        }
        RawTail::LoopReturn { theta } => {
            return Err(eyre!(
                "non-reconverging multi-exit loop (exits {:?}) inside a loop-demux arm is not \
                 handled",
                theta.exit_blocks
            ));
        }
    };
    Ok(LoopCaptureRegion { items, exit })
}

/// Walk the acyclic region starting at `start`, bounded by `boundary`, building
/// the role-agnostic item list and classifying the terminal branch/return into a
/// [`RawTail`]. Internal reconvergences (split-join and owning `p`-demux gammas,
/// nested loops) are emitted as items and the walk continues past them; the
/// region's exit-forming terminal is returned for the role producer to interpret.
///
/// `entry_prev` is the block control arrived from (to resolve a boundary's phis
/// when the region is empty). `loop_header`, when it equals `start`, lowers the
/// loop header as the first item rather than re-dispatching it as its own theta.
fn walk_items(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
    entry_prev: Option<BasicBlockId>,
    loop_header: Option<BasicBlockId>,
) -> color_eyre::Result<(Vec<RegionItem>, RawTail)> {
    let mut items: Vec<RegionItem> = Vec::new();
    let mut current = start;
    let mut prev = entry_prev;
    let mut first = true;

    loop {
        // When structuring a loop body, the header is the start: it is lowered
        // here (not treated as the repeat boundary or re-dispatched as its own
        // theta). Every subsequent block is checked normally, so the back-edge to
        // the header stops as a repeat and inner loops dispatch as nested thetas.
        let at_loop_header = first && loop_header == Some(current);
        first = false;

        if !at_loop_header {
            if boundary.contains(&current) || current == fn_ctx.exit_block_id {
                return Ok((
                    items,
                    RawTail::Continuation {
                        reached: current,
                        via: prev,
                    },
                ));
            }

            if let Some(scc_id) = fn_ctx.multi_entry_dispatch[current.0 as usize] {
                let arcs = &fn_ctx.scc_tree.arcs[scc_id.0 as usize];
                let entries: Vec<BasicBlockId> = arcs.entry_blocks.iter().copied().collect();
                let mut exit_targets: Vec<BasicBlockId> = Vec::new();
                for &(_, dest) in &arcs.exit_arcs {
                    if !exit_targets.contains(&dest) {
                        exit_targets.push(dest);
                    }
                }
                // Deterministic exit-vertex order (ascending block id), matching
                // the reducible path's `exit_blocks` ordering, so the exit `q` each
                // tail is keyed by is stable -- and so the exact exit edge can be
                // recovered when the RVSDG is later turned back into a CFG.
                exit_targets.sort_unstable_by_key(|block| block.0);
                if exit_targets.is_empty() {
                    return Err(eyre!(
                        "endless irreducible loop at block {} is not handled by the restructure \
                         transform",
                        current.0
                    ));
                }

                // Entry region: from the dispatch dominator `current` to the entry
                // vertices (which bound it), producing the entry `q` + entry-phi
                // inits. `current` is lowered here (loop_header marks it as the
                // start so the multi-entry dispatch is not re-detected).
                let entry_region = structure_entry(fn_ctx, current, &entries, Some(current))?;

                // One body per entry vertex: boundary = entries (repeat) ∪ exit
                // targets (exit).
                let mut body_boundary: SmallVec<[BasicBlockId; 8]> =
                    entries.iter().copied().collect();
                for &target in &exit_targets {
                    if !body_boundary.contains(&target) {
                        body_boundary.push(target);
                    }
                }
                let mut bodies: Vec<LoopBodyRegion> = Vec::with_capacity(entries.len());
                for &entry in &entries {
                    bodies.push(structure_loop_body(
                        fn_ctx,
                        entry,
                        &body_boundary,
                        None,
                        Some(entry),
                    )?);
                }

                let (exit_demux, exit_target) =
                    match structure_exit_demux(fn_ctx, &exit_targets, boundary, None)? {
                        ExitOutcome::Resume { target } => (None, target),
                        ExitOutcome::Reconverge { demux, join } => (Some(demux), join),
                        ExitOutcome::Return { .. } => {
                            return Err(eyre!(
                                "multi-entry loop at block {} with non-reconverging exits is not \
                                 yet handled",
                                current.0
                            ));
                        }
                    };

                items.push(RegionItem::Theta(ThetaNode {
                    scc: scc_id,
                    exit_blocks: exit_targets,
                    exit_demux,
                    kind: ThetaKind::MultiEntry {
                        entries,
                        entry_region,
                        bodies,
                    },
                }));
                prev = None;
                current = exit_target;
                continue;
            }

            if let Some(scc_id) = fn_ctx.scc_entry_block_to_id[current.0 as usize] {
                let arcs = &fn_ctx.scc_tree.arcs[scc_id.0 as usize];
                if arcs.entry_blocks.len() != 1 {
                    return Err(eyre!(
                        "multi-entry loop at block {} is not handled by the restructure transform",
                        current.0
                    ));
                }
                let header = arcs.entry_blocks[0];
                let exit_blocks: Vec<BasicBlockId> = arcs.exit_blocks.iter().copied().collect();
                let mut loop_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
                loop_boundary.push(header);
                loop_boundary.extend_from_slice(&exit_blocks);
                let body = structure_loop_body(fn_ctx, header, &loop_boundary, None, Some(header))?;

                // Endless loop (no exit arc): the theta never terminates, so
                // control diverges after it.
                if exit_blocks.is_empty() {
                    items.push(RegionItem::Theta(ThetaNode {
                        scc: scc_id,
                        exit_blocks: Vec::new(),
                        exit_demux: None,
                        kind: ThetaKind::Reducible { header, body },
                    }));
                    return Ok((items, RawTail::Diverge));
                }

                // Each exit tail is entered from its exit-arc source, so the
                // reconvergence's phis resolve along that edge -- including the case
                // where an exit arc targets the join directly.
                let arc_source = |exit_block: BasicBlockId| {
                    arcs.exit_arcs
                        .iter()
                        .find(|&&(_, dest)| dest == exit_block)
                        .map(|&(source, _)| source)
                };
                match structure_exit_demux(fn_ctx, &exit_blocks, boundary, Some(&arc_source))? {
                    ExitOutcome::Resume { target } => {
                        items.push(RegionItem::Theta(ThetaNode {
                            scc: scc_id,
                            exit_blocks,
                            exit_demux: None,
                            kind: ThetaKind::Reducible { header, body },
                        }));
                        prev = None;
                        current = target;
                        continue;
                    }
                    ExitOutcome::Reconverge { demux, join } => {
                        items.push(RegionItem::Theta(ThetaNode {
                            scc: scc_id,
                            exit_blocks,
                            exit_demux: Some(demux),
                            kind: ThetaKind::Reducible { header, body },
                        }));
                        prev = None;
                        current = join;
                        continue;
                    }
                    ExitOutcome::Return { demux } => {
                        // Every exit tail returns/diverges: the loop terminates
                        // the enclosing region (no block to resume at).
                        let theta = ThetaNode {
                            scc: scc_id,
                            exit_blocks,
                            exit_demux: Some(demux),
                            kind: ThetaKind::Reducible { header, body },
                        };
                        return Ok((items, RawTail::LoopReturn { theta }));
                    }
                }
            }
        }

        items.push(RegionItem::Block(current));

        let bb = &fn_ctx.func.basic_blocks[current.0 as usize];
        match &bb.term {
            llvm_ir::Terminator::Br(br) => {
                let next = *fn_ctx.bb_mapper.get_expect(&br.dest);
                prev = Some(current);
                current = next;
            }
            llvm_ir::Terminator::Ret(_) => {
                return Ok((items, RawTail::Return { block: current }));
            }
            llvm_ir::Terminator::Unreachable(_) => {
                return Ok((items, RawTail::Diverge));
            }
            llvm_ir::Terminator::CondBr(_) | llvm_ir::Terminator::Switch(_) => {
                let arm_targets = arm_target_blocks(fn_ctx, current)?;
                let continuations =
                    branch_continuation_points(fn_ctx, current, &arm_targets, boundary);
                let in_region = |block: BasicBlockId| {
                    block != fn_ctx.exit_block_id && !boundary.contains(&block)
                };
                let in_region_count = continuations
                    .iter()
                    .filter(|&&block| in_region(block))
                    .count();

                // The single join, if any: one in-region reconvergence (exactly
                // one continuation) or -- when no arm reconverges in-region -- the
                // post-dominator (one arm continues there, the others diverge). A
                // post-dominator of the synthetic exit means every arm
                // returns/diverges: a return gamma.
                let single_join: Option<BasicBlockId> = if continuations.is_empty() {
                    let join =
                        fn_ctx.post_immediate_dominators[current.0 as usize].ok_or_else(|| {
                            eyre!(
                                "non-reconverging branch at block {} has no post-dominator",
                                current.0
                            )
                        })?;
                    if join == fn_ctx.exit_block_id {
                        return Ok((
                            items,
                            RawTail::ReturnGamma {
                                head: current,
                                arm_targets,
                            },
                        ));
                    }
                    Some(join)
                } else if continuations.len() == 1 {
                    Some(continuations[0])
                } else {
                    None
                };

                if let Some(join) = single_join {
                    // A plain split-join gamma: arms reach `join` (resolving its
                    // phis) or diverge (poison). `join` may be in-region (the walk
                    // continues there) or a boundary block (the region exits there
                    // with its phis bound by the gamma).
                    let mut arm_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
                    arm_boundary.push(join);
                    arm_boundary.extend_from_slice(boundary);
                    let mut arms: Vec<SeqRegion> = Vec::with_capacity(arm_targets.len());
                    for &target in &arm_targets {
                        arms.push(structure_seq(
                            fn_ctx,
                            target,
                            &arm_boundary,
                            Some(current),
                            None,
                        )?);
                    }
                    items.push(RegionItem::Gamma(GammaNode {
                        head: current,
                        merge: GammaMerge::SingleJoin { join, arms },
                    }));
                    prev = None;
                    current = join;
                } else if in_region_count == 0 {
                    // Every continuation is an enclosing boundary: a pure router.
                    return Ok((
                        items,
                        RawTail::Route {
                            head: current,
                            arm_targets,
                        },
                    ));
                } else if in_region_count == continuations.len() {
                    // Every continuation is in-region: the owning `p`-demux.
                    let join =
                        fn_ctx.post_immediate_dominators[current.0 as usize].ok_or_else(|| {
                            eyre!(
                                "multi-continuation branch at block {} has no post-dominator",
                                current.0
                            )
                        })?;

                    let mut demux_targets: Vec<BasicBlockId> = continuations.to_vec();
                    if !demux_targets.contains(&join) {
                        demux_targets.push(join);
                    }
                    demux_targets.sort_unstable_by_key(|block| block.0);

                    // Head arms stop at any demux target (whichever they reach).
                    let mut head_boundary: SmallVec<[BasicBlockId; 8]> =
                        demux_targets.iter().copied().collect();
                    head_boundary.extend_from_slice(boundary);
                    let mut head_arms: Vec<CaptureRegion> = Vec::with_capacity(arm_targets.len());
                    for &target in &arm_targets {
                        head_arms.push(structure_capture(
                            fn_ctx,
                            target,
                            &head_boundary,
                            Some(current),
                        )?);
                    }

                    // Tails: lower each demux target's continuation to `join` once.
                    let mut demux_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
                    demux_boundary.push(join);
                    demux_boundary.extend_from_slice(boundary);
                    let mut tails: Vec<DemuxTail> = Vec::with_capacity(demux_targets.len());
                    for &target in &demux_targets {
                        if target == join {
                            tails.push(DemuxTail::Join);
                        } else {
                            let tail_boundary: SmallVec<[BasicBlockId; 8]> = demux_boundary
                                .iter()
                                .copied()
                                .filter(|&block| block != target)
                                .collect();
                            tails.push(DemuxTail::Tail(structure_seq(
                                fn_ctx,
                                target,
                                &tail_boundary,
                                None,
                                None,
                            )?));
                        }
                    }

                    items.push(RegionItem::Gamma(GammaNode {
                        head: current,
                        merge: GammaMerge::Demux {
                            head_arms,
                            spec: DemuxSpec {
                                demux_targets,
                                join,
                                tails,
                            },
                        },
                    }));
                    prev = None;
                    current = join;
                } else {
                    // Mixed: some continuations are in-region, some are boundaries.
                    // Only a loop body lowers this; the role producer builds it.
                    return Ok((
                        items,
                        RawTail::MixedDemux {
                            head: current,
                            arm_targets,
                            continuations,
                        },
                    ));
                }
            }
            other => {
                return Err(eyre!(
                    "unsupported terminator at block {} in the restructure transform: {:?}",
                    current.0,
                    other
                ));
            }
        }
    }
}

/// Decide the post-theta exit handling shared by both loop kinds. A single exit
/// vertex resumes directly ([`ExitOutcome::Resume`]). Multiple exit vertices that
/// reconverge at one `join` form an exit-`q` demux whose `tails[i]` lower
/// `exit_blocks[i]` to `join` ([`ExitOutcome::Reconverge`]). Multiple exit
/// vertices whose tails all return/diverge form a terminal return demux
/// ([`ExitOutcome::Return`]) -- the loop ends the enclosing region. `arc_source`,
/// when given, maps an exit vertex to the exit-arc source the tail is entered
/// from (so the tail resolves a join's phis along that edge).
fn structure_exit_demux(
    fn_ctx: &FnCtx,
    exit_blocks: &[BasicBlockId],
    boundary: &[BasicBlockId],
    arc_source: Option<&dyn Fn(BasicBlockId) -> Option<BasicBlockId>>,
) -> color_eyre::Result<ExitOutcome> {
    if exit_blocks.len() == 1 {
        return Ok(ExitOutcome::Resume {
            target: exit_blocks[0],
        });
    }
    let entry_prev = |exit_block| arc_source.and_then(|lookup| lookup(exit_block));

    let continuations = continuation_points(fn_ctx, exit_blocks, boundary);
    if let [join] = continuations.as_slice() {
        let join = *join;
        let mut tail_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
        tail_boundary.push(join);
        tail_boundary.extend_from_slice(boundary);
        let mut tails: Vec<SeqRegion> = Vec::with_capacity(exit_blocks.len());
        for &exit_block in exit_blocks {
            tails.push(structure_seq(
                fn_ctx,
                exit_block,
                &tail_boundary,
                entry_prev(exit_block),
                None,
            )?);
        }
        return Ok(ExitOutcome::Reconverge {
            demux: ExitDemux {
                tails,
                merge: ExitMerge::Reconverge { join },
            },
            join,
        });
    }

    // No single reconvergence. The tails may still each leave on their own (every
    // one returning or diverging), in which case the loop is the enclosing
    // region's terminal: a return gamma on the exit `q`. The tails are structured
    // against the enclosing boundary, since they share no join.
    let mut tails: Vec<SeqRegion> = Vec::with_capacity(exit_blocks.len());
    for &exit_block in exit_blocks {
        tails.push(structure_seq(
            fn_ctx,
            exit_block,
            boundary,
            entry_prev(exit_block),
            None,
        )?);
    }
    let all_terminal = tails.iter().all(|tail| {
        matches!(
            tail.exit,
            SeqExit::Return { .. } | SeqExit::Diverge | SeqExit::ReturnGamma { .. }
        )
    });
    if all_terminal {
        Ok(ExitOutcome::Return {
            demux: ExitDemux {
                tails,
                merge: ExitMerge::Return,
            },
        })
    } else {
        Err(eyre!(
            "multi-exit loop whose {} exit vertices have {} reconvergence points is not handled \
             by the restructure transform",
            exit_blocks.len(),
            continuations.len()
        ))
    }
}
