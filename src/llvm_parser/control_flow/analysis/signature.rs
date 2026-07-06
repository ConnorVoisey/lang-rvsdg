//! Phi lookups shared by arc-payload application and the emitter: the
//! leading phi run of a block, and a phi's incoming value for a particular
//! predecessor. Phi destinations are bound when an arc is traversed (they
//! are the paper's copies-on-arcs), never lowered as instructions.

use llvm_ir::{BasicBlock, Name, Operand, instruction::Phi};
use smallvec::SmallVec;

use crate::llvm_parser::block_mapper::BasicBlockId;

/// The leading run of phi instructions at the start of `bb`. In LLVM every
/// phi precedes the first non-phi instruction of its block, so a
/// `take_while` over that prefix collects them all.
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
/// predecessor. Used to pick the value a phi takes along a particular CFG
/// edge.
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
