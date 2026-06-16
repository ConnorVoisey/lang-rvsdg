//! Phi-driven signature analysis (Bahmann, Reissmann, Jahre, Meyer 2015,
//! section 4), SSA-native. The gamma/theta signature is read **directly off the
//! LLVM phi nodes** -- no liveness fixpoint, no write-sets -- and a region's
//! live-ins come from one operand scan over the *walked* block set (never a
//! dominator set).
//!
//! These are free functions (taking `&FnCtx` and, where needed, the symbol
//! table) so the restructuring transform and the construction walk share one
//! implementation.

use llvm_ir::{BasicBlock, Name, Operand, instruction::Phi};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::BasicBlockId,
        instructions::{for_each_operand, instruction_dest},
    },
    rvsdg::ValueId,
};

/// The leading run of phi instructions at the start of `bb`. In LLVM every phi
/// precedes the first non-phi instruction of its block, so a `take_while` over
/// that prefix collects them all.
pub(in crate::llvm_parser) fn phi_instructions_at(bb: &BasicBlock) -> SmallVec<[&Phi; 4]> {
    bb.instrs
        .iter()
        .map_while(|inst| match inst {
            llvm_ir::Instruction::Phi(phi) => Some(phi),
            _ => None,
        })
        .collect()
}

/// The incoming `(value, predecessor-name)` of `phi` whose predecessor block
/// satisfies `pred_matches`, or `None` if no incoming arm comes from such a
/// predecessor. Used to pick the value a phi takes along a particular CFG edge.
pub(in crate::llvm_parser) fn phi_incoming_from<'a>(
    phi: &'a Phi,
    bb_mapper: &crate::llvm_parser::block_mapper::BasicBlockMapper,
    pred_matches: impl Fn(BasicBlockId) -> bool,
) -> Option<(&'a Operand, &'a Name)> {
    phi.incoming_values.iter().find_map(|(operand, pred_name)| {
        let pred_id = bb_mapper.get(pred_name)?;
        pred_matches(*pred_id).then_some((operand, pred_name))
    })
}

/// The set of blocks a region walk actually covers from `start`: every block
/// reachable by following CFG successors, stopping at (and not crossing) any
/// `boundary` block or the synthetic function exit. This is the *walked region*
/// -- the block set whose SSA uses-minus-defs are the region's live-ins
/// (`region_live_ins`). An empty arm (its `start` already a boundary) walks
/// nothing.
#[tracing::instrument(skip_all)]
pub(in crate::llvm_parser) fn collect_walked_blocks(
    fn_ctx: &FnCtx,
    start: BasicBlockId,
    boundary: &[BasicBlockId],
) -> FxHashSet<BasicBlockId> {
    let exit = fn_ctx.exit_block_id;
    let stops = |block: BasicBlockId| block == exit || boundary.contains(&block);
    let mut walked = FxHashSet::default();
    let mut stack: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
    if !stops(start) {
        stack.push(start);
    }
    while let Some(block) = stack.pop() {
        if !walked.insert(block) {
            continue;
        }
        for &succ in fn_ctx.bb_mapper.outputs(block) {
            if !stops(succ) && !walked.contains(&succ) {
                stack.push(succ);
            }
        }
    }
    walked
}

/// A region's **live-ins**: SSA values *used* inside the walked block set but
/// *defined outside* it -- the gamma/theta value inputs (Bahmann et al. section
/// 4, read off the phi nodes). Returns parallel vectors so the caller can seed
/// each subregion's `name -> param(i)` map: `names[i]`'s outer value is
/// `values[i]`.
///
/// Two passes: (1) collect names defined inside (instruction phi dests); (2) scan
/// every instruction operand -- any `LocalOperand` not defined inside, resolvable
/// in the outer-scope `name_to_value`, is a live-in. Then the join phi operands
/// flowing in from a walked block (or directly along the `pass_through_pred ->
/// join` edge for an empty arm) are added the same way. SSA dominance guarantees
/// a used value's def precedes its uses, so resolution through `name_to_value` is
/// exact.
///
/// `pass_through_pred` is the branch head whose `head -> join` edge an empty arm
/// passes a join-phi value along (an outer-scope value threaded in as a live-in
/// so the arm can echo it straight back out); `None` when no such pass-through
/// edge applies.
#[tracing::instrument(skip_all)]
pub(in crate::llvm_parser) fn region_live_ins(
    fn_ctx: &FnCtx,
    name_to_value: &FxHashMap<Name, ValueId>,
    walked_blocks: &[BasicBlockId],
    phis_at_join: &[&Phi],
    pass_through_pred: Option<BasicBlockId>,
) -> (Vec<Name>, Vec<ValueId>) {
    // Pass 1: names defined inside the walked region, plus an O(1)-membership
    // set of the walked blocks themselves (used by the join-phi scan below
    // instead of a linear `walked_blocks.contains`, which would be O(phis x
    // incomings x walked) over a potentially large region).
    let mut defined_inside: FxHashSet<&Name> = FxHashSet::default();
    let mut walked_set: FxHashSet<BasicBlockId> =
        FxHashSet::with_capacity_and_hasher(walked_blocks.len(), Default::default());
    for &bb_id in walked_blocks {
        walked_set.insert(bb_id);
        let bb = &fn_ctx.func.basic_blocks[bb_id.0 as usize];
        for inst in &bb.instrs {
            if let Some(dest) = instruction_dest(inst) {
                defined_inside.insert(dest);
            }
        }
    }

    // Pass 2: operands used inside but not defined inside, resolvable outside.
    // `seen` dedups; `names`/`values` are the parallel output vectors.
    let mut seen: FxHashSet<Name> = FxHashSet::default();
    let mut names: Vec<Name> = Vec::new();
    let mut values: Vec<ValueId> = Vec::new();

    // Records `name` as a live-in if it is not defined inside, not already seen,
    // and resolves in the outer-scope `name_to_value`. The accumulators are
    // passed as `&mut` params rather than captured so the scan loops below can
    // still borrow them between calls.
    let push_live_in = |name: &Name,
                        seen: &mut FxHashSet<Name>,
                        names: &mut Vec<Name>,
                        values: &mut Vec<ValueId>| {
        if defined_inside.contains(name) || !seen.insert(name.clone()) {
            return;
        }
        if let Some(&value) = name_to_value.get(name) {
            names.push(name.clone());
            values.push(value);
        }
    };

    for &bb_id in walked_blocks {
        let bb = &fn_ctx.func.basic_blocks[bb_id.0 as usize];
        for inst in &bb.instrs {
            for_each_operand(inst, |operand| {
                if let Operand::LocalOperand { name, .. } = operand {
                    push_live_in(name, &mut seen, &mut names, &mut values);
                }
            });
        }
        // Terminator operands are uses too (a branch condition, a switch
        // selector, or a `ret` value). A returned value used only in the
        // terminator must still be threaded in -- this matters for the
        // non-reconverging / early-return branch path.
        match &bb.term {
            llvm_ir::Terminator::Ret(ret) => {
                if let Some(Operand::LocalOperand { name, .. }) = &ret.return_operand {
                    push_live_in(name, &mut seen, &mut names, &mut values);
                }
            }
            llvm_ir::Terminator::CondBr(cb) => {
                if let Operand::LocalOperand { name, .. } = &cb.condition {
                    push_live_in(name, &mut seen, &mut names, &mut values);
                }
            }
            llvm_ir::Terminator::Switch(sw) => {
                if let Operand::LocalOperand { name, .. } = &sw.operand {
                    push_live_in(name, &mut seen, &mut names, &mut values);
                }
            }
            _ => {}
        }
    }

    // Join phi operands: each arm contributes one per phi, resolved inside the arm,
    // so its name must be seeded as a live-in. Accept operands from a walked
    // block or along the pass-through head -> join edge.
    for phi in phis_at_join {
        for (operand, pred_name) in &phi.incoming_values {
            let Some(&pred_id) = fn_ctx.bb_mapper.get(pred_name) else {
                continue;
            };
            if !walked_set.contains(&pred_id) && Some(pred_id) != pass_through_pred {
                continue;
            }
            if let Operand::LocalOperand { name, .. } = operand {
                push_live_in(name, &mut seen, &mut names, &mut values);
            }
        }
    }

    (names, values)
}
