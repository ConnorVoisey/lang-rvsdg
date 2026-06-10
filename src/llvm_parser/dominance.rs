use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockInOuts};

trait CfgView {
    fn entry(&self) -> BasicBlockId;
    fn block_count(&self) -> usize;
    fn predecessors(&self, bb_id: BasicBlockId) -> &[BasicBlockId];
    fn successors(&self, bb_id: BasicBlockId) -> &[BasicBlockId];
}
pub struct ForwardView<'a> {
    pub nodes: &'a [BasicBlockInOuts],
    pub entry: BasicBlockId,
}
pub struct ReverseView<'a> {
    pub nodes: &'a [BasicBlockInOuts],
    pub exit: BasicBlockId,
}

impl<'a> CfgView for ForwardView<'a> {
    fn entry(&self) -> BasicBlockId {
        self.entry
    }

    fn block_count(&self) -> usize {
        self.nodes.len()
    }

    fn predecessors(&self, bb_id: BasicBlockId) -> &[BasicBlockId] {
        &self.nodes[bb_id.0 as usize].inputs
    }

    fn successors(&self, bb_id: BasicBlockId) -> &[BasicBlockId] {
        &self.nodes[bb_id.0 as usize].outputs
    }
}
impl<'a> CfgView for ReverseView<'a> {
    fn entry(&self) -> BasicBlockId {
        self.exit
    }

    fn block_count(&self) -> usize {
        self.nodes.len()
    }

    fn predecessors(&self, bb_id: BasicBlockId) -> &[BasicBlockId] {
        &self.nodes[bb_id.0 as usize].outputs
    }

    fn successors(&self, bb_id: BasicBlockId) -> &[BasicBlockId] {
        &self.nodes[bb_id.0 as usize].inputs
    }
}

fn get_reverse_post_order<V: CfgView>(view: &V) -> Vec<BasicBlockId> {
    let mut visited = vec![false; view.block_count()];
    let mut ordered = Vec::with_capacity(view.block_count());
    #[inline(always)]
    fn dfs<V: CfgView>(
        id: BasicBlockId,
        view: &V,
        visited: &mut Vec<bool>,
        order: &mut Vec<BasicBlockId>,
    ) {
        if visited[id.0 as usize] {
            return;
        }
        visited[id.0 as usize] = true;

        for output in view.successors(id) {
            dfs(*output, view, visited, order);
        }
        order.push(id);
    }

    dfs(view.entry(), view, &mut visited, &mut ordered);

    // probably don't need to actually reverse the order, could just use it in reverse.
    ordered.reverse();
    ordered
}

/// Compute the immediate-dominator tree for `view`, indexed by block id.
///
/// The returned vector has one entry per block. A block's entry is:
///   - `Some(idom)` for blocks reachable from the view's entry,
///   - `Some(self)` for the entry block itself,
///   - `None` for blocks NOT reachable from the view's entry.
///
/// For a [`ReverseView`] (post-dominators) the "entry" is the function
/// exit, so the unreachable blocks are exactly those that cannot reach the
/// exit: arms ending in a `noreturn` call, `unreachable`, or an infinite
/// loop. Those have no post-dominator, hence `None`. Callers that need a
/// post-dominator at a branch read it for the branch block, which (because
/// at least one arm reaches the exit) is itself reachable and so resolves.
pub fn compute_dominance<V: CfgView>(view: &V) -> Vec<Option<BasicBlockId>> {
    let reverse_post_order = get_reverse_post_order(view);
    compute_dominance_with_order(view, &reverse_post_order)
}

fn compute_dominance_with_order<V: CfgView>(
    view: &V,
    reverse_post_order: &[BasicBlockId],
) -> Vec<Option<BasicBlockId>> {
    // `reverse_post_order` covers only the blocks reachable from the view's
    // entry; blocks that cannot reach it are absent and keep `idom = None`
    // below. The iterative solver and `intersect` only ever touch RPO
    // blocks (a non-RPO block has `idom = None`, so it is skipped as a
    // predecessor and never walked into), so a partial order is sound.
    debug_assert!(reverse_post_order.len() <= view.block_count());

    // immediate_dominators[bb_id.0 as usize] = the immediate dominator of block bb_id,
    // or None if it hasn't been computed yet.
    let mut immediate_dominators = vec![None; view.block_count()];
    let start_node = reverse_post_order[0];
    immediate_dominators[start_node.0 as usize] = Some(start_node);

    let reverse_post_order_indexes = {
        let mut reverse_post_order_indexes = vec![0; view.block_count()];
        for (i, bb_id) in reverse_post_order.iter().enumerate() {
            reverse_post_order_indexes[bb_id.0 as usize] = i;
        }
        reverse_post_order_indexes
    };

    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..reverse_post_order.len() {
            let bb_id = reverse_post_order[i];
            let predecessors = view.predecessors(bb_id);
            let mut new_imediate_dominator: Option<BasicBlockId> = None;
            for predecessor in predecessors {
                if immediate_dominators[predecessor.0 as usize].is_some() {
                    match new_imediate_dominator {
                        Some(new_i_dom) => {
                            new_imediate_dominator = Some(intersect(
                                *predecessor,
                                new_i_dom,
                                &immediate_dominators,
                                &reverse_post_order_indexes,
                            ))
                        }
                        None => new_imediate_dominator = Some(*predecessor),
                    }
                }
            }

            if new_imediate_dominator.is_some()
                && immediate_dominators[bb_id.0 as usize] != new_imediate_dominator
            {
                immediate_dominators[bb_id.0 as usize] = new_imediate_dominator;
                changed = true;
            }
        }
    }
    immediate_dominators
}

/// find where two nodes parents intersect
fn intersect(
    b1: BasicBlockId,
    b2: BasicBlockId,
    immediate_dominators: &[Option<BasicBlockId>],
    reverse_post_order: &[usize],
) -> BasicBlockId {
    let mut finger_1 = b1;
    let mut finger_2 = b2;

    // Since we have the reverse post order, we can simply keep walking until finger 1 eq finger 2.
    // While finger 1s post order is lower than finger 2, walk up finger 1.
    // Then do the same for finger 2 until they intersect at a node
    while finger_1 != finger_2 {
        while reverse_post_order[finger_1.0 as usize] > reverse_post_order[finger_2.0 as usize] {
            finger_1 = immediate_dominators[finger_1.0 as usize]
                .expect("immediate domiantor should have been calculated");
        }
        while reverse_post_order[finger_2.0 as usize] > reverse_post_order[finger_1.0 as usize] {
            finger_2 = immediate_dominators[finger_2.0 as usize]
                .expect("immediate domiantor should have been calculated");
        }
    }

    finger_1
}

/// Returns true if block `d` dominates block `b`. Walks the dominator-tree
/// chain from `b` upward via `immediate_dominators`; returns true on hitting
/// `d`, false on hitting the entry (self-idom) without finding `d`.
///
/// Every block dominates itself.
#[inline(always)]
pub fn dominates(
    d: BasicBlockId,
    b: BasicBlockId,
    immediate_dominators: &[Option<BasicBlockId>],
) -> bool {
    let mut current = b;
    loop {
        if current == d {
            return true;
        }
        match immediate_dominators[current.0 as usize] {
            Some(parent) if parent == current => return false,
            Some(parent) => current = parent,
            None => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::test_utils::{idoms_of, init, post_idoms_of};
    use pretty_assertions::assert_eq;

    // ------------------------------------------------------------------------
    // Trivial cases
    // ------------------------------------------------------------------------

    #[test]
    fn single_block_dominates_itself() {
        // Just the entry, no edges.
        let (mapper, bbs) = init(1);
        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
    }

    #[test]
    fn linear_chain() {
        // 0 → 1 → 2 → 3
        // Each block's idom is its only predecessor.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[1]), idoms[2]);
        assert_eq!(Some(bbs[2]), idoms[3]);
    }

    // ------------------------------------------------------------------------
    // Branching (γ-style)
    // ------------------------------------------------------------------------

    #[test]
    fn symmetric_diamond() {
        //     0
        //    / \
        //   1   2
        //    \ /
        //     3
        // Join block 3 is dominated by 0 (not 1 or 2 — either could be skipped).
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[3]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[0]), idoms[2]);
        assert_eq!(Some(bbs[0]), idoms[3]); // join dominated by entry, not 1 or 2
    }

    #[test]
    fn triangle_early_exit() {
        //    0
        //   / \
        //  1   |
        //   \ /
        //    2
        // 0 → 1 → 2 AND 0 → 2. Block 2 is dominated by 0, not 1.
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[2]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[0]), idoms[2]);
    }

    #[test]
    fn multi_arm_switch() {
        //      0
        //    / | \
        //   1  2  3
        //    \ | /
        //      4
        // 3-way split joining at block 4. All arms dominated by 0; join also.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[0], bbs[3]);
        mapper.add_connection(bbs[1], bbs[4]);
        mapper.add_connection(bbs[2], bbs[4]);
        mapper.add_connection(bbs[3], bbs[4]);

        let idoms = idoms_of(&mapper);
        for &b in &bbs {
            assert_eq!(Some(bbs[0]), idoms[b.0 as usize]);
        }
    }

    #[test]
    fn nested_diamonds() {
        //       0
        //      / \
        //     1   2
        //      \ /
        //       3      ← idom 0
        //      / \
        //     4   5
        //      \ /
        //       6      ← idom 3
        let (mut mapper, bbs) = init(7);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);
        mapper.add_connection(bbs[3], bbs[5]);
        mapper.add_connection(bbs[4], bbs[6]);
        mapper.add_connection(bbs[5], bbs[6]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[0]), idoms[2]);
        assert_eq!(Some(bbs[0]), idoms[3]); // first join
        assert_eq!(Some(bbs[3]), idoms[4]);
        assert_eq!(Some(bbs[3]), idoms[5]);
        assert_eq!(Some(bbs[3]), idoms[6]); // second join
    }

    #[test]
    fn asymmetric_branch_lengths() {
        //     0
        //    / \
        //   1   2
        //   |   |
        //   3   |
        //    \ /
        //     4
        // The left arm has one extra block. Join is still dominated by 0.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);
        mapper.add_connection(bbs[2], bbs[4]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[0]), idoms[2]);
        assert_eq!(Some(bbs[1]), idoms[3]); // sole pred is 1
        assert_eq!(Some(bbs[0]), idoms[4]); // join dominated by entry
    }

    // ------------------------------------------------------------------------
    // Loops (θ-style)
    // ------------------------------------------------------------------------

    #[test]
    fn simple_self_loop() {
        // 0 → 1 → 1 (self), 1 → 2
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[1]), idoms[2]); // exit dominated by header
    }

    #[test]
    fn natural_loop() {
        //   0 (preheader)
        //   ↓
        //   1 ←─ 2  (header ← latch)
        //   ↓
        //   3 (exit)
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]); // back-edge
        mapper.add_connection(bbs[1], bbs[3]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]); // header dominated by preheader
        assert_eq!(Some(bbs[1]), idoms[2]); // latch dominated by header
        assert_eq!(Some(bbs[1]), idoms[3]); // exit dominated by header
    }

    #[test]
    fn loop_with_internal_branch() {
        // Shape your 04_while_loop.c.ll produces:
        //   0 (preheader)
        //   ↓
        //   1 (header, tests cond)
        //  ↙ ↘
        // 3   2 (exit)
        // └→ 1 (back-edge: 3 → 1)
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]); // header → latch (continue)
        mapper.add_connection(bbs[1], bbs[2]); // header → exit
        mapper.add_connection(bbs[3], bbs[1]); // back-edge

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]); // header dominated by preheader
        assert_eq!(Some(bbs[1]), idoms[2]); // exit dominated by header
        assert_eq!(Some(bbs[1]), idoms[3]); // latch dominated by header
    }

    #[test]
    fn loop_with_break() {
        // Shape your 05_loop_with_break.c.ll has, simplified:
        //   0 (preheader)
        //   ↓
        //   1 (header) ─── exit ─→ 4 (post-loop)
        //   ↓                       ↑
        //   2 (body)─── break ──────┘
        //   ↓
        //   3 (latch) → 1 (back-edge)
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[1], bbs[4]); // exit at header
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[2], bbs[4]); // break: exit at body
        mapper.add_connection(bbs[3], bbs[1]); // back-edge

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]); // header
        assert_eq!(Some(bbs[1]), idoms[2]); // body dominated by header
        assert_eq!(Some(bbs[2]), idoms[3]); // latch dominated by body
        assert_eq!(Some(bbs[1]), idoms[4]); // post-loop dominated by header
    }

    #[test]
    fn nested_loops() {
        // Outer loop {1, 2, 3, 4, 5}, inner loop {2, 3}.
        //   0 (preheader)
        //   ↓
        //   1 (outer header) ←─── 5 (outer latch)
        //   ↓
        //   2 (inner header) ←─ 3 (inner latch)
        //   ↓
        //   4 (after inner)
        //   ↓
        //   5 → 1 (outer back-edge)
        //   plus 1 → 6 (final exit when outer cond is false)
        let (mut mapper, bbs) = init(7);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[1], bbs[6]); // outer exit
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[2], bbs[4]); // inner exit
        mapper.add_connection(bbs[3], bbs[2]); // inner back-edge
        mapper.add_connection(bbs[4], bbs[5]);
        mapper.add_connection(bbs[5], bbs[1]); // outer back-edge

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]); // outer header
        assert_eq!(Some(bbs[1]), idoms[2]); // inner header dominated by outer header
        assert_eq!(Some(bbs[2]), idoms[3]); // inner latch dominated by inner header
        assert_eq!(Some(bbs[2]), idoms[4]); // after-inner dominated by inner header
        assert_eq!(Some(bbs[4]), idoms[5]); // outer latch dominated by after-inner
        assert_eq!(Some(bbs[1]), idoms[6]); // final exit dominated by outer header
    }

    // ------------------------------------------------------------------------
    // Tricky cases
    // ------------------------------------------------------------------------

    #[test]
    fn dom_chain_walks_to_entry() {
        // Verify the IDom chain from a deep block walks all the way to the entry.
        //   0 → 1 → 2 → 3 → 4
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);

        let idoms = idoms_of(&mapper);

        // Walk from block 4 up to the entry via idom chain.
        let mut chain = vec![bbs[4]];
        let mut current = bbs[4];
        loop {
            let parent = idoms[current.0 as usize].unwrap();
            if parent == current {
                break;
            }
            chain.push(parent);
            current = parent;
        }
        assert_eq!(vec![bbs[4], bbs[3], bbs[2], bbs[1], bbs[0]], chain);
    }

    #[test]
    fn cross_edge_dominance() {
        // 0 → 1, 0 → 2, 1 → 2 (cross edge from one arm of a branch to the other)
        //   0
        //  / \
        // 1   |
        //  \  v
        //   → 2
        // Block 2 has two predecessors (0 and 1); dominated by 0.
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[2]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[0]), idoms[2]);
    }

    #[test]
    fn long_chain_then_branch() {
        // Linear chain feeding into a branch.
        // 0 → 1 → 2 → 3 (branches) → 4, 3 → 5, 4 → 6, 5 → 6
        let (mut mapper, bbs) = init(7);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);
        mapper.add_connection(bbs[3], bbs[5]);
        mapper.add_connection(bbs[4], bbs[6]);
        mapper.add_connection(bbs[5], bbs[6]);

        let idoms = idoms_of(&mapper);
        assert_eq!(Some(bbs[0]), idoms[0]);
        assert_eq!(Some(bbs[0]), idoms[1]);
        assert_eq!(Some(bbs[1]), idoms[2]);
        assert_eq!(Some(bbs[2]), idoms[3]);
        assert_eq!(Some(bbs[3]), idoms[4]);
        assert_eq!(Some(bbs[3]), idoms[5]);
        assert_eq!(Some(bbs[3]), idoms[6]); // join dominated by branch point
    }

    // ========================================================================
    // Post-dominator tests
    //
    // Reminder: B post-dominates A iff every path from A to the function exit
    // passes through B. The exit block post-dominates itself; everything else
    // is post-dominated by something closer to the exit than itself.
    // ========================================================================

    #[test]
    fn post_dom_single_block() {
        // Single block IS the exit; post-dominates itself.
        let (mapper, bbs) = init(1);
        let pdoms = post_idoms_of(&mapper, bbs[0]);
        assert_eq!(Some(bbs[0]), pdoms[0]);
    }

    #[test]
    fn post_dom_block_that_cannot_reach_exit_is_none() {
        // 0 → 1 → 3 (exit)
        // 0 → 2 → 2 (self-loop: block 2 never reaches the exit)
        // Block 2 is unreachable in the reverse CFG, so it has no
        // post-dominator (None) rather than tripping a precondition. The
        // branch at 0 still resolves: its only exit-reaching path is via 1.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[2]);

        let pdoms = post_idoms_of(&mapper, bbs[3]);
        assert_eq!(Some(bbs[3]), pdoms[3]); // exit post-dominates itself
        assert_eq!(Some(bbs[3]), pdoms[1]); // 1 → 3
        assert_eq!(Some(bbs[1]), pdoms[0]); // only exit path is 0 → 1 → 3
        assert_eq!(None, pdoms[2]); // 2 cannot reach the exit
    }

    #[test]
    fn post_dom_linear_chain() {
        // 0 → 1 → 2 → 3 (exit)
        // Each block's post-idom is its only successor.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);

        let pdoms = post_idoms_of(&mapper, bbs[3]);
        assert_eq!(Some(bbs[3]), pdoms[3]); // exit post-dominates itself
        assert_eq!(Some(bbs[3]), pdoms[2]); // 2 → 3
        assert_eq!(Some(bbs[2]), pdoms[1]); // 1 → 2
        assert_eq!(Some(bbs[1]), pdoms[0]); // 0 → 1
    }

    #[test]
    fn post_dom_diamond() {
        //     0
        //    / \
        //   1   2
        //    \ /
        //     3 (exit)
        // Every path from any block ends at 3 — all post-dominated by 3.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[3]);

        let pdoms = post_idoms_of(&mapper, bbs[3]);
        assert_eq!(Some(bbs[3]), pdoms[3]);
        assert_eq!(Some(bbs[3]), pdoms[2]);
        assert_eq!(Some(bbs[3]), pdoms[1]);
        assert_eq!(Some(bbs[3]), pdoms[0]); // branch source also post-dominated by join
    }

    #[test]
    fn post_dom_triangle_early_exit() {
        //    0
        //   / \
        //  1   |
        //   \ /
        //    2 (exit)
        // The path 0 → 2 bypasses 1, so 1 does NOT post-dominate 0.
        // But 2 post-dominates everything.
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[2]);

        let pdoms = post_idoms_of(&mapper, bbs[2]);
        assert_eq!(Some(bbs[2]), pdoms[2]);
        assert_eq!(Some(bbs[2]), pdoms[1]); // 1's only successor is 2
        assert_eq!(Some(bbs[2]), pdoms[0]); // both branches end at 2
    }

    #[test]
    fn post_dom_multi_arm_switch() {
        //      0
        //    / | \
        //   1  2  3
        //    \ | /
        //      4 (exit)
        // All arms reconverge at 4.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[0], bbs[3]);
        mapper.add_connection(bbs[1], bbs[4]);
        mapper.add_connection(bbs[2], bbs[4]);
        mapper.add_connection(bbs[3], bbs[4]);

        let pdoms = post_idoms_of(&mapper, bbs[4]);
        for &b in &bbs {
            assert_eq!(Some(bbs[4]), pdoms[b.0 as usize]);
        }
    }

    #[test]
    fn post_dom_nested_diamonds() {
        //       0
        //      / \
        //     1   2
        //      \ /
        //       3      ← post-idom of 1 and 2 (and 0)
        //      / \
        //     4   5
        //      \ /
        //       6 (exit)   ← post-idom of 3 (and 4 and 5)
        let (mut mapper, bbs) = init(7);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);
        mapper.add_connection(bbs[3], bbs[5]);
        mapper.add_connection(bbs[4], bbs[6]);
        mapper.add_connection(bbs[5], bbs[6]);

        let pdoms = post_idoms_of(&mapper, bbs[6]);
        assert_eq!(Some(bbs[6]), pdoms[6]);
        assert_eq!(Some(bbs[6]), pdoms[5]); // 5 → 6
        assert_eq!(Some(bbs[6]), pdoms[4]); // 4 → 6
        assert_eq!(Some(bbs[6]), pdoms[3]); // 3's arms reconverge at 6
        assert_eq!(Some(bbs[3]), pdoms[2]); // 2 → 3 (only successor)
        assert_eq!(Some(bbs[3]), pdoms[1]); // 1 → 3 (only successor)
        assert_eq!(Some(bbs[3]), pdoms[0]); // 0's arms reconverge at 3, which precedes the second diamond
    }

    #[test]
    fn post_dom_asymmetric_branch_lengths() {
        //     0
        //    / \
        //   1   2
        //   |   |
        //   3   |
        //    \ /
        //     4 (exit)
        // Both arms converge at 4, but 3 only post-dominates 1.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);
        mapper.add_connection(bbs[2], bbs[4]);

        let pdoms = post_idoms_of(&mapper, bbs[4]);
        assert_eq!(Some(bbs[4]), pdoms[4]);
        assert_eq!(Some(bbs[4]), pdoms[3]); // 3 → 4
        assert_eq!(Some(bbs[4]), pdoms[2]); // 2 → 4
        assert_eq!(Some(bbs[3]), pdoms[1]); // 1 → 3 (sole succ)
        assert_eq!(Some(bbs[4]), pdoms[0]); // arms reconverge at 4
    }

    #[test]
    fn post_dom_natural_loop() {
        //   0 (preheader)
        //   ↓
        //   1 ←─ 2  (header ← latch)
        //   ↓
        //   3 (exit)
        // post-idom[0] = 1 (preheader's only successor is header)
        // post-idom[1] = 3 (header eventually exits to 3)
        // post-idom[2] = 1 (latch's only successor is header)
        // post-idom[3] = 3 (exit post-dominates itself)
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]); // back-edge
        mapper.add_connection(bbs[1], bbs[3]);

        let pdoms = post_idoms_of(&mapper, bbs[3]);
        assert_eq!(Some(bbs[3]), pdoms[3]);
        assert_eq!(Some(bbs[3]), pdoms[1]); // header eventually reaches exit
        assert_eq!(Some(bbs[1]), pdoms[2]); // latch's only successor is header
        assert_eq!(Some(bbs[1]), pdoms[0]); // preheader's only successor is header
    }

    #[test]
    fn post_dom_chain_walks_to_exit() {
        // Verify post-idom chain from block 0 walks to the exit.
        //   0 → 1 → 2 → 3 → 4 (exit)
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);

        let pdoms = post_idoms_of(&mapper, bbs[4]);

        let mut chain = vec![bbs[0]];
        let mut current = bbs[0];
        loop {
            let parent = pdoms[current.0 as usize].unwrap();
            if parent == current {
                break;
            }
            chain.push(parent);
            current = parent;
        }
        assert_eq!(vec![bbs[0], bbs[1], bbs[2], bbs[3], bbs[4]], chain);
    }

    #[test]
    fn post_dom_self_loop() {
        // 0 → 1 → 1 (self), 1 → 2 (exit)
        // post-idom[0] = 1 (sole successor)
        // post-idom[1] = 2 (eventually exits to 2)
        // post-idom[2] = 2
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);

        let pdoms = post_idoms_of(&mapper, bbs[2]);
        assert_eq!(Some(bbs[2]), pdoms[2]);
        assert_eq!(Some(bbs[2]), pdoms[1]);
        assert_eq!(Some(bbs[1]), pdoms[0]);
    }

    // ========================================================================
    // Dom/post-dom symmetry — these structural invariants tie the two together.
    // ========================================================================

    #[test]
    fn dom_and_post_dom_diamond_symmetry() {
        // In a symmetric diamond, the entry dominates the join, AND the join
        // post-dominates the entry. The "branch point" of dominators is the
        // "merge point" of post-dominators — they're structurally mirrored.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[3]);

        let idoms = idoms_of(&mapper);
        let pdoms = post_idoms_of(&mapper, bbs[3]);

        // Forward: everything dominated by entry.
        for &b in &bbs {
            assert_eq!(Some(bbs[0]), idoms[b.0 as usize]);
        }
        // Backward: everything post-dominated by exit.
        for &b in &bbs {
            assert_eq!(Some(bbs[3]), pdoms[b.0 as usize]);
        }
    }

    // ------------------------------------------------------------------------
    // dominates() function — regression coverage for the multi-hop bug
    // (was indexing `b.0` instead of `current.0` and never moved up the chain).
    // ------------------------------------------------------------------------

    #[test]
    fn dominates_self() {
        // Trivial: any block dominates itself.
        let (mut mapper, bbs) = init(2);
        mapper.add_connection(bbs[0], bbs[1]);
        let idoms = idoms_of(&mapper);
        assert!(dominates(bbs[0], bbs[0], &idoms));
        assert!(dominates(bbs[1], bbs[1], &idoms));
    }

    #[test]
    fn dominates_immediate() {
        // 0 -> 1. idom(1) = 0. 0 dominates 1, 1 does NOT dominate 0.
        let (mut mapper, bbs) = init(2);
        mapper.add_connection(bbs[0], bbs[1]);
        let idoms = idoms_of(&mapper);
        assert!(dominates(bbs[0], bbs[1], &idoms));
        assert!(!dominates(bbs[1], bbs[0], &idoms));
    }

    #[test]
    fn dominates_chain() {
        // 0 -> 1 -> 2 -> 3. Pre-fix this was the failing case: dominates(0, 2)
        // and dominates(0, 3) would walk one step (to idom(2)=1, idom(3)=2),
        // then re-read idom(b) instead of idom(current), get the same answer,
        // see parent==current, and return false.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        let idoms = idoms_of(&mapper);
        // Every block dominates itself.
        for &b in &bbs {
            assert!(dominates(b, b, &idoms));
        }
        // Each block dominates everything below it in the chain.
        for i in 0..bbs.len() {
            for j in i..bbs.len() {
                assert!(
                    dominates(bbs[i], bbs[j], &idoms),
                    "block {} should dominate block {}",
                    i,
                    j
                );
            }
        }
        // Blocks lower in the chain do NOT dominate higher ones.
        for i in 0..bbs.len() {
            for j in 0..i {
                assert!(
                    !dominates(bbs[i], bbs[j], &idoms),
                    "block {} should NOT dominate block {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn dominates_sibling_not_in_chain() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3. Diamond.
        // 1 and 2 are siblings in the dominator tree (both directly under 0);
        // neither dominates the other.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[3]);
        let idoms = idoms_of(&mapper);
        assert!(!dominates(bbs[1], bbs[2], &idoms));
        assert!(!dominates(bbs[2], bbs[1], &idoms));
    }

    #[test]
    fn dominates_entry_dominates_all() {
        // Entry block dominates every reachable block in any CFG.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[1], bbs[4]);
        let idoms = idoms_of(&mapper);
        for &b in &bbs {
            assert!(
                dominates(bbs[0], b, &idoms),
                "entry should dominate block {}",
                b.0
            );
        }
    }
}
