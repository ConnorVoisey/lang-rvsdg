//! Per-SCC arc/vertex sets used by loop restructuring (§4.1 of Bahmann et al. 2014).
//!
//! Given an SCC, this computes the five sets the restructuring rewrite needs:
//! - entry blocks: blocks inside the SCC reached from outside it
//! - entry arcs: the corresponding incoming edges
//! - exit blocks: blocks outside the SCC reached from inside it
//! - exit arcs: the corresponding outgoing edges
//! - repetition arcs: edges inside the SCC that target an entry block (back-edges)
//!
//! `is_natural` is the cheap test for "this loop already has a single header,
//! single back-edge, and single exit" — restructuring can be skipped.

use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    strongly_connected_components::Scc,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoopArcs {
    pub entry_blocks: SmallVec<[BasicBlockId; 4]>,
    pub entry_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]>,
    pub exit_blocks: SmallVec<[BasicBlockId; 4]>,
    pub exit_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]>,
    pub repetition_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]>,
}

impl LoopArcs {
    pub fn from_scc(scc: &Scc, mapper: &BasicBlockMapper) -> Self {
        let block_count = mapper.blocks.len();
        let mut in_scc = vec![false; block_count];
        for &b in &scc.blocks {
            in_scc[b.0 as usize] = true;
        }

        let mut is_entry = vec![false; block_count];
        let mut is_exit_target = vec![false; block_count];
        let mut entry_arcs = SmallVec::new();
        let mut exit_arcs = SmallVec::new();

        for &block in &scc.blocks {
            for &pred in mapper.inputs(block) {
                if !in_scc[pred.0 as usize] {
                    entry_arcs.push((pred, block));
                    is_entry[block.0 as usize] = true;
                }
            }
            for &succ in mapper.outputs(block) {
                if !in_scc[succ.0 as usize] {
                    exit_arcs.push((block, succ));
                    is_exit_target[succ.0 as usize] = true;
                }
            }
        }

        let mut repetition_arcs = SmallVec::new();
        for &block in &scc.blocks {
            for &succ in mapper.outputs(block) {
                if is_entry[succ.0 as usize] {
                    repetition_arcs.push((block, succ));
                }
            }
        }

        let mut entry_blocks = (0..block_count as u32)
            .filter(|i| is_entry[*i as usize])
            .map(BasicBlockId)
            .collect::<SmallVec<_>>();
        let mut exit_blocks = (0..block_count as u32)
            .filter(|i| is_exit_target[*i as usize])
            .map(BasicBlockId)
            .collect::<SmallVec<_>>();
        entry_blocks.sort();
        exit_blocks.sort();

        Self {
            entry_blocks,
            entry_arcs,
            exit_blocks,
            exit_arcs,
            repetition_arcs,
        }
    }

    /// One header, one back-edge, one exit — the canonical do/while shape that
    /// maps directly onto a θ-node.
    #[inline]
    pub fn is_natural(&self) -> bool {
        self.entry_blocks.len() == 1 && self.repetition_arcs.len() == 1 && self.exit_arcs.len() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::strongly_connected_components::Scc;
    use llvm_ir::Name;
    use pretty_assertions::assert_eq;

    fn init(n: usize) -> (BasicBlockMapper, Vec<BasicBlockId>) {
        let mut mapper = BasicBlockMapper::new(n);
        let bbs = (0..n)
            .map(|i| mapper.intern(&Name::Number(i)))
            .collect::<Vec<_>>();
        (mapper, bbs)
    }

    fn scc_of(blocks: &[BasicBlockId]) -> Scc {
        Scc {
            blocks: blocks.iter().copied().collect(),
            is_trivial: false,
        }
    }

    #[test]
    fn natural_two_block_loop() {
        // 0 (preheader) -> 1 (header) -> 2 (body) -> 1, 1 -> 3 (exit)
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);

        let scc = scc_of(&[bbs[1], bbs[2]]);
        let mut arcs = LoopArcs::from_scc(&scc, &mapper);
        arcs.entry_arcs.sort();
        arcs.exit_arcs.sort();
        arcs.repetition_arcs.sort();

        assert_eq!(vec![bbs[1]], arcs.entry_blocks.to_vec());
        assert_eq!(vec![(bbs[0], bbs[1])], arcs.entry_arcs.to_vec());
        assert_eq!(vec![bbs[3]], arcs.exit_blocks.to_vec());
        assert_eq!(vec![(bbs[1], bbs[3])], arcs.exit_arcs.to_vec());
        assert_eq!(vec![(bbs[2], bbs[1])], arcs.repetition_arcs.to_vec());
        assert!(arcs.is_natural());
    }

    #[test]
    fn single_block_self_loop() {
        // 0 (entry) -> 1, 1 -> 1 (self), 1 -> 2 (exit)
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);

        let scc = scc_of(&[bbs[1]]);
        let arcs = LoopArcs::from_scc(&scc, &mapper);

        assert_eq!(vec![bbs[1]], arcs.entry_blocks.to_vec());
        assert_eq!(vec![(bbs[0], bbs[1])], arcs.entry_arcs.to_vec());
        assert_eq!(vec![bbs[2]], arcs.exit_blocks.to_vec());
        assert_eq!(vec![(bbs[1], bbs[2])], arcs.exit_arcs.to_vec());
        assert_eq!(vec![(bbs[1], bbs[1])], arcs.repetition_arcs.to_vec());
        assert!(arcs.is_natural());
    }

    #[test]
    fn irreducible_two_entries() {
        // 0 enters both 1 and 2; 1 <-> 2 form a cycle; 1 -> 3 exits.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[0], bbs[2]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);

        let scc = scc_of(&[bbs[1], bbs[2]]);
        let mut arcs = LoopArcs::from_scc(&scc, &mapper);
        arcs.entry_arcs.sort();
        arcs.exit_arcs.sort();
        arcs.repetition_arcs.sort();

        assert_eq!(vec![bbs[1], bbs[2]], arcs.entry_blocks.to_vec());
        assert_eq!(
            vec![(bbs[0], bbs[1]), (bbs[0], bbs[2])],
            arcs.entry_arcs.to_vec()
        );
        assert_eq!(vec![bbs[3]], arcs.exit_blocks.to_vec());
        assert_eq!(vec![(bbs[1], bbs[3])], arcs.exit_arcs.to_vec());
        // Both intra-SCC edges target entry blocks → both are repetition arcs.
        assert_eq!(
            vec![(bbs[1], bbs[2]), (bbs[2], bbs[1])],
            arcs.repetition_arcs.to_vec()
        );
        assert!(!arcs.is_natural());
    }

    #[test]
    fn multiple_exits_not_natural() {
        // 0 -> 1, 1 -> 2 -> 1 (loop), 1 -> 3 (exit a), 2 -> 4 (exit b)
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[2], bbs[4]);

        let scc = scc_of(&[bbs[1], bbs[2]]);
        let mut arcs = LoopArcs::from_scc(&scc, &mapper);
        arcs.exit_arcs.sort();

        assert_eq!(vec![bbs[1]], arcs.entry_blocks.to_vec());
        assert_eq!(vec![bbs[3], bbs[4]], arcs.exit_blocks.to_vec());
        assert_eq!(
            vec![(bbs[1], bbs[3]), (bbs[2], bbs[4])],
            arcs.exit_arcs.to_vec()
        );
        assert_eq!(vec![(bbs[2], bbs[1])], arcs.repetition_arcs.to_vec());
        assert!(!arcs.is_natural());
    }

    #[test]
    fn multiple_back_edges_not_natural() {
        // 0 -> 1, 1 -> 2 -> 1, 1 -> 3 -> 1 (two latches), 1 -> 4 (exit)
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);
        mapper.add_connection(bbs[3], bbs[1]);
        mapper.add_connection(bbs[1], bbs[4]);

        let scc = scc_of(&[bbs[1], bbs[2], bbs[3]]);
        let mut arcs = LoopArcs::from_scc(&scc, &mapper);
        arcs.repetition_arcs.sort();

        assert_eq!(vec![bbs[1]], arcs.entry_blocks.to_vec());
        assert_eq!(vec![bbs[4]], arcs.exit_blocks.to_vec());
        assert_eq!(
            vec![(bbs[2], bbs[1]), (bbs[3], bbs[1])],
            arcs.repetition_arcs.to_vec()
        );
        assert!(!arcs.is_natural());
    }
}
