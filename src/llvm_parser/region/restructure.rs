//! Phase A — branch restructuring analysis (Bahmann et al. 2015 §4.2).
//!
//! Pure analysis: for the branch terminating a block, compute its
//! *continuation points* — the tail blocks where the fan-out arms rejoin.
//! This is the only datum Phase B (the build) needs to lower a branch
//! without ever lowering a block twice:
//!
//!   - exactly one continuation point → a plain single-join γ;
//!   - more than one                  → a `p`-demux γ at the single tail
//!     (the auxiliary continuation predicate `p`).
//!
//! No graph is materialised — the result is a small set of existing
//! `BasicBlockId`s ("just enough to say there is a predicate here"). The
//! `p` value itself is discovered during the build (which continuation an
//! arm reaches), not stored here.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::llvm_parser::{FnCtx, block_mapper::BasicBlockId};

/// The continuation points of the branch terminating `head`, within the
/// acyclic region bounded by `boundary` (the blocks where the enclosing
/// region ends — reaching one stops the walk; it belongs to the enclosing
/// level, not this branch). The synthetic function exit always stops.
///
/// §4.2: a continuation point is a tail block where the region's `entries`
/// reconverge. We classify arm membership by **region-local reachability**,
/// not global dominance: each entry is BFS-ed within the region, tagging
/// every block it reaches with a bit. A block reached from two or more
/// entries is a merge; the merge blocks that are themselves an entry (an
/// empty arm / shared fan-out target) or are entered from a strictly-fewer-
/// entry predecessor are the continuation points. (Global dominance is wrong
/// here: a block reached from one entry *and* from outside the region — a
/// cross edge into an enclosing branch's tail — is dominated by neither yet
/// is not where *these* entries rejoin.)
///
/// `entries` are the region's fan-out targets (a branch's arm targets, or —
/// for the recursive demux and the loop body — a synthesized set of
/// continuation/entry blocks). `boundary` is where the enclosing region ends
/// (reaching it stops the walk); the synthetic function exit always stops.
///
/// Exactly one continuation point → a plain single-join γ; more than one →
/// the `p`-demux. Returned in ascending block-id order for deterministic
/// lowering.
pub(super) fn continuation_points(
    fn_ctx: &FnCtx,
    entries: &[BasicBlockId],
    boundary: &[BasicBlockId],
) -> SmallVec<[BasicBlockId; 4]> {
    let exit = fn_ctx.exit_block_id;
    let stops = |b: BasicBlockId| b == exit || boundary.contains(&b);

    // arm_reach[v] = the set of entries (by index, as a bitmask) from which
    // `v` is reachable without crossing the boundary. One BFS per entry;
    // entries past 63 share the top bit (harmless over-approximation — at
    // worst it under-reports a merge and we fall back to the post-dominator).
    let mut arm_reach: FxHashMap<BasicBlockId, u64> = FxHashMap::default();
    for (j, &start) in entries.iter().enumerate() {
        if stops(start) {
            continue;
        }
        let bit = 1u64 << j.min(63);
        let mut stack: SmallVec<[BasicBlockId; 16]> = SmallVec::new();
        stack.push(start);
        while let Some(b) = stack.pop() {
            let mask = arm_reach.entry(b).or_insert(0);
            if *mask & bit != 0 {
                continue; // already tagged for this entry
            }
            *mask |= bit;
            for &succ in fn_ctx.bb_mapper.outputs(b) {
                if !stops(succ) {
                    stack.push(succ);
                }
            }
        }
    }

    // A continuation point is a merge block (reached from >= 2 entries) that
    // is itself an entry (a shared fan-out target) or is entered from a
    // predecessor reached by a different (so strictly fewer) set of entries —
    // i.e. the entry to the merged region, not a block already interior to it.
    let mut continuation_points: SmallVec<[BasicBlockId; 4]> = SmallVec::new();
    for (&v, &mask) in &arm_reach {
        if mask.count_ones() < 2 {
            continue;
        }
        let is_merge_entry = entries.contains(&v)
            || fn_ctx
                .bb_mapper
                .inputs(v)
                .iter()
                .any(|&p| arm_reach.get(&p).is_some_and(|&pm| pm != mask));
        if is_merge_entry {
            continuation_points.push(v);
        }
    }
    continuation_points.sort_unstable_by_key(|b| b.0);
    continuation_points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::region::test_fixture::TestFn;

    fn cps(test_fn: &TestFn, head: &str) -> Vec<BasicBlockId> {
        let fn_ctx = test_fn.fn_ctx();
        let entries: Vec<BasicBlockId> = fn_ctx.bb_mapper.outputs(test_fn.block(head)).to_vec();
        continuation_points(&fn_ctx, &entries, &[])
            .into_iter()
            .collect()
    }

    #[test]
    fn diamond_has_one_continuation_point() {
        // entry -> t -> j
        //       -> f -> j
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
  %r = phi i32 [ %a, %t ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        assert_eq!(cps(&test_fn, "entry"), vec![test_fn.block("j")]);
    }

    #[test]
    fn arm_with_interior_block_still_one_continuation_point() {
        // entry -> t -> mid -> j
        //       -> f -> j
        // `mid` is interior to the true arm; only `j` is a continuation.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %mid
mid:
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %mid ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        assert_eq!(cps(&test_fn, "entry"), vec![test_fn.block("j")]);
        // `mid` is not a continuation point (single predecessor in one arm).
        assert!(!cps(&test_fn, "entry").contains(&test_fn.block("mid")));
    }

    #[test]
    fn switch_fallthrough_has_multiple_continuation_points() {
        // A switch where case 0 falls through into case 1's block, and case
        // 1 / default merge at j. `c1` (shared by the case-1 arc and the
        // case-0 fall-through) and `j` are both continuation points -> the
        // build needs a p-demux.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %x, i32 %a) {
entry:
  switch i32 %x, label %def [ i32 0, label %c0
                              i32 1, label %c1 ]
c0:
  br label %c1
c1:
  br label %j
def:
  br label %j
j:
  ret i32 %a
}
"#,
        );
        let got = cps(&test_fn, "entry");
        assert!(
            got.contains(&test_fn.block("c1")),
            "c1 is a fall-through continuation"
        );
        assert!(got.contains(&test_fn.block("j")), "j is the final merge");
        assert!(
            !got.contains(&test_fn.block("c0")),
            "c0 is an arm root, not a continuation"
        );
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn empty_arm_continuation_is_a_shared_successor() {
        // entry -> j         (true arm empty: straight to the join)
        //       -> f -> j
        // `j` is a fan-out target reached from `head` AND from arm `f`, so
        // it is a shared continuation, not an arm root.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %j, label %f
f:
  br label %j
j:
  %r = phi i32 [ %a, %entry ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        assert_eq!(cps(&test_fn, "entry"), vec![test_fn.block("j")]);
    }

    #[test]
    fn both_arms_return_has_no_continuation_point() {
        // Neither arm reconverges inside the region (both return); the only
        // shared point is the function exit, which is not a continuation.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  ret i32 %a
f:
  ret i32 %b
}
"#,
        );
        assert!(cps(&test_fn, "entry").is_empty());
    }
}
