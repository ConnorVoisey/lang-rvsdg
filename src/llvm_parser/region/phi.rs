//! Shared phi-instruction helpers used by both the loop and branch
//! lowering paths. Lives at the region-module level rather than inside
//! either sub-module because both need to peek into phi structure
//! without depending on each other.

use llvm_ir::{Instruction, Name, Operand, instruction::Phi};
use smallvec::SmallVec;

use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockMapper};

/// Return the leading run of `Phi` instructions at the start of
/// `basic_block`. LLVM IR requires all phis to appear contiguously at
/// the start of a block; we stop at the first non-phi.
pub(super) fn phi_instructions_at(
    basic_block: &llvm_ir::BasicBlock,
) -> SmallVec<[&llvm_ir::instruction::Phi; 4]> {
    basic_block
        .instrs
        .iter()
        .map_while(|i| match i {
            Instruction::Phi(p) => Some(p),
            _ => None,
        })
        .collect()
}

/// Find the incoming pair of `phi` whose predecessor block satisfies
/// `is_in_set`. Used by:
///
///   - `arm_phi_contributions`: the predecessor lying in the current
///     gamma-arm's block set tells us which incoming this arm
///     contributes.
///   - The loop-closed-phi classification in `analyze_loop`: the
///     predecessor lying inside the loop body tells us which incoming
///     the loop produces.
///
/// Returns `None` if no incoming matches (caller decides whether that
/// is an error or just a "this phi isn't ours").
///
/// `is_in_set` is taken as a closure rather than a slice/set so callers
/// can choose the membership representation (linear scan over a small
/// SmallVec for typical loops; FxHashSet for larger sets) without
/// forcing one shape on this helper.
pub(super) fn phi_incoming_from<'a>(
    phi: &'a Phi,
    bb_mapper: &BasicBlockMapper,
    is_in_set: impl Fn(BasicBlockId) -> bool,
) -> Option<(&'a Operand, &'a Name)> {
    phi.incoming_values
        .iter()
        .find_map(|(op, pred_name)| match bb_mapper.get(pred_name) {
            Some(&pred_id) if is_in_set(pred_id) => Some((op, pred_name)),
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::region::test_fixture::{TestFn, local_name};
    use pretty_assertions::assert_eq;

    #[test]
    fn phi_instructions_at_returns_phis_at_block_start() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %r1 = phi i32 [ %a, %t ], [ %b, %f ]
  %r2 = phi i32 [ %b, %t ], [ %a, %f ]
  %r3 = add i32 %r1, %r2
  ret i32 %r3
}
"#,
        );
        let j = test_fn.block("j");
        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        assert_eq!(phis.len(), 2);
        assert_eq!(phis[0].dest, local_name("r1"));
        assert_eq!(phis[1].dest, local_name("r2"));
    }

    #[test]
    fn phi_instructions_at_empty_when_no_phi() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a) {
entry:
  ret i32 %a
}
"#,
        );
        let entry = test_fn.block("entry");
        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[entry.0 as usize]);
        assert!(phis.is_empty());
    }

    #[test]
    fn phi_instructions_at_stops_at_first_non_phi() {
        // Only the phi prefix is taken; if a non-phi sits between phis
        // (which is invalid LLVM but we should still defend), we stop
        // early.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %r1 = phi i32 [ %a, %t ], [ %b, %f ]
  %s = add i32 %r1, 1
  ret i32 %s
}
"#,
        );
        let j = test_fn.block("j");
        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        assert_eq!(phis.len(), 1);
        assert_eq!(phis[0].dest, local_name("r1"));
    }
}
