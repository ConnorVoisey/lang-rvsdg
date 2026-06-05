//! Per-component arc and vertex sets, used by the paper's loop
//! restructuring algorithm (Bahmann, Reissmann, Jahre, Meyer (2015)
//! "Perfect Reconstructability of Control Flow from Demand Dependence
//! Graphs", section 4.1).
//!
//! Given a block set that represents a loop's body (a strongly
//! connected component, or a synthesised candidate produced during
//! restructuring), this computes the five sets the restructuring
//! rewrite needs:
//!
//! - **entry blocks** (paper's V_E): blocks inside the body that are
//!   reached from outside it.
//! - **entry arcs** (A_E): the corresponding incoming edges.
//! - **exit blocks** (V_X): blocks outside the body that are reached
//!   from inside it.
//! - **exit arcs** (A_X): the corresponding outgoing edges.
//! - **repetition arcs** (A_R): edges inside the body whose target is
//!   an entry block; the back-edges that close the loop.
//!
//! The analysis is shape-agnostic. It does not assume the block set
//! forms a natural loop. Multi-entry and multi-exit subsets work
//! identically; the paper's q and r predicates use these sets to
//! produce a single-entry single-exit shape suitable for direct
//! conversion into a theta node. See `scc_tree.rs` for how loop
//! restructuring builds and consumes the analysis.

use smallvec::SmallVec;

use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockMapper};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopArcs {
    /// Blocks inside the loop body that are reached from outside it.
    pub entry_blocks: SmallVec<[BasicBlockId; 4]>,
    /// Edges (from outside, to inside) that target an entry block.
    pub entry_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]>,
    /// Blocks outside the loop body that are reached from inside it.
    pub exit_blocks: SmallVec<[BasicBlockId; 4]>,
    /// Edges (from inside, to outside) that leave the loop.
    pub exit_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]>,
    /// Edges inside the loop whose target is an entry block; the
    /// back-edges that close the loop.
    pub repetition_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]>,
}

impl LoopArcs {
    /// Compute the arc and vertex sets for an arbitrary block set viewed
    /// as a loop body. Given a strongly connected component's blocks,
    /// produces the entry blocks (V_E), entry arcs (A_E), exit blocks
    /// (V_X), exit arcs (A_X), and repetition arcs (A_R) that the
    /// paper's section 4.1 restructuring consumes.
    pub fn from_scc_blocks(blocks: &[BasicBlockId], mapper: &BasicBlockMapper) -> Self {
        let block_count = mapper.blocks.len();
        let mut in_body = vec![false; block_count];
        for &b in blocks {
            in_body[b.0 as usize] = true;
        }

        let mut is_entry = vec![false; block_count];
        let mut is_exit_target = vec![false; block_count];
        let mut entry_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]> = SmallVec::new();
        let mut exit_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]> = SmallVec::new();

        for &block in blocks {
            for &pred in mapper.inputs(block) {
                if !in_body[pred.0 as usize] {
                    entry_arcs.push((pred, block));
                    is_entry[block.0 as usize] = true;
                }
            }
            for &succ in mapper.outputs(block) {
                if !in_body[succ.0 as usize] {
                    exit_arcs.push((block, succ));
                    is_exit_target[succ.0 as usize] = true;
                }
            }
        }

        let mut repetition_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]> = SmallVec::new();
        for &block in blocks {
            for &succ in mapper.outputs(block) {
                if is_entry[succ.0 as usize] {
                    repetition_arcs.push((block, succ));
                }
            }
        }

        let mut entry_blocks: SmallVec<[BasicBlockId; 4]> = (0..block_count as u32)
            .filter(|i| is_entry[*i as usize])
            .map(BasicBlockId)
            .collect();
        let mut exit_blocks: SmallVec<[BasicBlockId; 4]> = (0..block_count as u32)
            .filter(|i| is_exit_target[*i as usize])
            .map(BasicBlockId)
            .collect();
        entry_blocks.sort();
        exit_blocks.sort();

        LoopArcs {
            entry_blocks,
            entry_arcs,
            exit_blocks,
            exit_arcs,
            repetition_arcs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::test_utils::init;
    use pretty_assertions::assert_eq;

    #[test]
    fn natural_two_block_loop() {
        // 0 (preheader) -> 1 (header) -> 2 (body) -> 1, 1 -> 3 (exit).
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);

        let arcs = LoopArcs::from_scc_blocks(&[bbs[1], bbs[2]], &mapper);

        assert_eq!(arcs.entry_blocks.to_vec(), vec![bbs[1]]);
        assert_eq!(arcs.entry_arcs.to_vec(), vec![(bbs[0], bbs[1])]);
        assert_eq!(arcs.exit_blocks.to_vec(), vec![bbs[3]]);
        assert_eq!(arcs.exit_arcs.to_vec(), vec![(bbs[1], bbs[3])]);
        assert_eq!(arcs.repetition_arcs.to_vec(), vec![(bbs[2], bbs[1])]);
    }

    #[test]
    fn single_block_self_loop() {
        // 0 (entry) -> 1, 1 -> 1 (self), 1 -> 2 (exit).
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);

        let arcs = LoopArcs::from_scc_blocks(&[bbs[1]], &mapper);

        // Header and latch coincide; the self-edge is the back-edge.
        assert_eq!(arcs.entry_blocks.to_vec(), vec![bbs[1]]);
        assert_eq!(arcs.entry_arcs.to_vec(), vec![(bbs[0], bbs[1])]);
        assert_eq!(arcs.exit_blocks.to_vec(), vec![bbs[2]]);
        assert_eq!(arcs.exit_arcs.to_vec(), vec![(bbs[1], bbs[2])]);
        assert_eq!(arcs.repetition_arcs.to_vec(), vec![(bbs[1], bbs[1])]);
    }

    #[test]
    fn multiple_exits_single_entry() {
        // 0 -> 1, 1 -> 2 -> 1 (loop), 1 -> 3 (exit a), 2 -> 4 (exit b).
        // Single-entry, multi-exit: the analysis records both exit arcs
        // and the single entry vertex unchanged.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[4]);

        let arcs = LoopArcs::from_scc_blocks(&[bbs[1], bbs[2]], &mapper);
        let mut exit_arcs = arcs.exit_arcs.to_vec();
        exit_arcs.sort();

        assert_eq!(arcs.entry_blocks.to_vec(), vec![bbs[1]]);
        assert_eq!(arcs.entry_arcs.to_vec(), vec![(bbs[0], bbs[1])]);
        assert_eq!(arcs.exit_blocks.to_vec(), vec![bbs[3], bbs[4]]);
        assert_eq!(exit_arcs, vec![(bbs[1], bbs[3]), (bbs[2], bbs[4])]);
        assert_eq!(arcs.repetition_arcs.to_vec(), vec![(bbs[2], bbs[1])]);
    }

    #[test]
    fn multiple_back_edges_single_exit() {
        // 0 -> 1, 1 -> 2 -> 1, 1 -> 3 -> 1 (two latches), 1 -> 4 (exit).
        // Multi-latch shapes show up as two repetition arcs to the same
        // single entry vertex.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[3], bbs[1]);
        mapper.add_connection(bbs[1], bbs[4]);

        let arcs = LoopArcs::from_scc_blocks(&[bbs[1], bbs[2], bbs[3]], &mapper);
        let mut rep = arcs.repetition_arcs.to_vec();
        rep.sort();

        assert_eq!(arcs.entry_blocks.to_vec(), vec![bbs[1]]);
        assert_eq!(arcs.entry_arcs.to_vec(), vec![(bbs[0], bbs[1])]);
        assert_eq!(arcs.exit_blocks.to_vec(), vec![bbs[4]]);
        assert_eq!(arcs.exit_arcs.to_vec(), vec![(bbs[1], bbs[4])]);
        assert_eq!(rep, vec![(bbs[2], bbs[1]), (bbs[3], bbs[1])]);
    }

    #[test]
    fn multi_entry_records_both_entry_vertices() {
        // 0 -> 1 and 0 -> 2; 1 -> 2 and 2 -> 1 form an irreducible
        // cycle; 1 -> 3 exits. The block set {1, 2} has two entry
        // vertices: this is the case that drives the paper's q
        // predicate.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);

        let arcs = LoopArcs::from_scc_blocks(&[bbs[1], bbs[2]], &mapper);
        let mut entry_arcs = arcs.entry_arcs.to_vec();
        entry_arcs.sort();

        assert_eq!(arcs.entry_blocks.to_vec(), vec![bbs[1], bbs[2]]);
        assert_eq!(entry_arcs, vec![(bbs[0], bbs[1]), (bbs[0], bbs[2])]);
        assert_eq!(arcs.exit_blocks.to_vec(), vec![bbs[3]]);
        assert_eq!(arcs.exit_arcs.to_vec(), vec![(bbs[1], bbs[3])]);
        let mut rep = arcs.repetition_arcs.to_vec();
        rep.sort();
        assert_eq!(rep, vec![(bbs[1], bbs[2]), (bbs[2], bbs[1])]);
    }
}
