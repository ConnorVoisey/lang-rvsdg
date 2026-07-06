//! Strongly connected component tree: a precomputed nesting hierarchy
//! of every non-trivial cycle in a function, with the entry / exit /
//! repetition arc analysis attached to each node.
//!
//! # What this is for
//!
//! Bahmann, Reissmann, Jahre, Meyer (2015) "Perfect Reconstructability
//! of Control Flow from Demand Dependence Graphs" section 4.1 turns
//! each strongly connected component into a theta node by introducing
//! auxiliary entry and repetition predicates. The algorithm is
//! recursive: after the outer component is restructured, the loop body
//! is itself a closed control flow graph that may still contain inner
//! cycles, and the same algorithm runs on it.
//!
//! Doing the recursion lazily during lowering would require either
//! mutating the control flow graph (to remove repetition arcs as we
//! descend) or threading restructuring state through every recursive
//! lowering call. We do neither. Instead we run the recursion once at
//! function setup and store the resulting tree on the function context.
//! Lowering reads from the tree by index; no Tarjan invocation happens
//! during lowering itself.
//!
//! # What the tree contains
//!
//! Every node represents one non-trivial strongly connected component
//! (trivial single-block components without self-loops are not loops
//! and are excluded). Each node stores:
//!
//! - The component's block set, exactly as Tarjan discovered it at the
//!   nesting level the node sits at.
//! - The entry, exit, and repetition arc analysis (see `LoopArcs`).
//! - Pointers to the node's parent (the strongly connected component
//!   one nesting level out, if any) and children (the strongly
//!   connected components one nesting level in, after this node's
//!   repetition arcs are removed).
//!
//! Outer nodes' block sets include their inner nodes' blocks; an inner
//! node is a subset of its parent. This intentional duplication keeps
//! lookups O(1) per node and avoids reconstructing the block set during
//! lowering. The total memory across all nodes is bounded by
//! O(block count * nesting depth).
//!
//! # Performance
//!
//! Build cost: one whole-function Tarjan plus one sub-Tarjan per
//! non-trivial component, each running over that component's block
//! set with repetition arcs filtered. Total work is bounded by
//! O((blocks + edges) * nesting depth). For typical function nesting
//! depths (under five), this is a small constant factor over the
//! single whole-function Tarjan we already pay for today.
//!
//! Lookups (block to component, component to children, component to
//! arcs) are all O(1) array indexing.
//!
//! TODO: this is more of an mvp than a production version,
//! there is far too much cloning everywhere.

use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    scc::arcs::LoopArcs,
};

/// Index into the SCC tree's node arrays. Distinct from the strongly
/// connected component identifiers produced by `SccAnalysis`: the tree
/// assigns its own indices in build order and includes inner components
/// that the whole-function Tarjan does not expose separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SccTreeNodeId(pub u32);

#[derive(Debug, Clone)]
pub struct SccTree {
    /// Identifiers of the components at the outermost nesting level
    /// (those with no enclosing component in the same function).
    pub roots: Vec<SccTreeNodeId>,
    /// Block set for each component, exactly as discovered by Tarjan at
    /// the nesting level the component sits at. An outer component's
    /// block set is a superset of every nested component's block set.
    pub blocks: Vec<SmallVec<[BasicBlockId; 8]>>,
    /// Direct children of each component: the components found by
    /// running Tarjan on this component's block set with its repetition
    /// arcs filtered out. Empty if the component has no nested loops.
    pub children: Vec<SmallVec<[SccTreeNodeId; 2]>>,
    /// Direct parent of each component, or None for root components.
    pub parent: Vec<Option<SccTreeNodeId>>,
    /// Entry, exit, and repetition arc analysis for each component.
    /// Computed once during build; reused by lowering.
    pub arcs: Vec<LoopArcs>,
}

impl SccTree {
    /// Build the tree for `mapper`. Performs one whole-function Tarjan
    /// pass to find the outermost components and then, for each
    /// non-trivial component, performs a sub-Tarjan over that
    /// component's blocks with the repetition arcs removed to discover
    /// inner components. Trivial components (single block with no
    /// self-loop) are dropped at every level; they are not loops.
    #[tracing::instrument(name = "SccTree::build", skip_all)]
    pub fn build(mapper: &BasicBlockMapper) -> Self {
        let mut tree = Self {
            roots: Vec::new(),
            blocks: Vec::new(),
            children: Vec::new(),
            parent: Vec::new(),
            arcs: Vec::new(),
        };

        // Outer Tarjan finds the function's top-level strongly connected
        // components. Trivial ones (singletons without self-loops) are
        // skipped at the top level just as they are at every level.
        let top_level = mapper.get_strongly_connected_components();
        for scc in &top_level.sccs {
            if scc.is_trivial {
                continue;
            }
            let blocks: SmallVec<[BasicBlockId; 8]> = scc.blocks.iter().copied().collect();
            let id = tree.alloc_node(blocks, None);
            tree.roots.push(id);
            tree.populate(id, mapper);
        }

        tree
    }

    /// Allocate a fresh tree node holding `blocks` with the given
    /// parent. Initialises `arcs` to an empty default; `populate`
    /// fills it in immediately after allocation.
    fn alloc_node(
        &mut self,
        blocks: SmallVec<[BasicBlockId; 8]>,
        parent: Option<SccTreeNodeId>,
    ) -> SccTreeNodeId {
        let id = SccTreeNodeId(self.blocks.len() as u32);
        self.blocks.push(blocks);
        self.children.push(SmallVec::new());
        self.parent.push(parent);
        // Placeholder; replaced by populate before any caller reads it.
        self.arcs.push(LoopArcs {
            entry_blocks: SmallVec::new(),
            entry_arcs: SmallVec::new(),
            exit_blocks: SmallVec::new(),
            exit_arcs: SmallVec::new(),
            repetition_arcs: SmallVec::new(),
        });
        id
    }

    /// Compute `arcs[node]` and recurse into the component's loop body
    /// to discover nested components. Called once per node, immediately
    /// after the node is allocated by `alloc_node`.
    fn populate(&mut self, node: SccTreeNodeId, mapper: &BasicBlockMapper) {
        // Take the blocks slice out by clone so the subsequent
        // mutations on `self` do not conflict with the borrow.
        let blocks = self.blocks[node.0 as usize].clone();

        let arcs = LoopArcs::from_scc_blocks(&blocks, mapper);
        // To find the nested loops, re-run Tarjan over this component's blocks
        // with the back-edges (repetition arcs) removed: without them the
        // component is no longer a single cycle, so any cycle that remains is a
        // genuinely nested loop. (This is the paper's L* construction, section
        // 4.1, done without mutating the actual control flow graph.)
        let repetition_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]> =
            arcs.repetition_arcs.clone();
        self.arcs[node.0 as usize] = arcs;

        let inner_sccs = mapper.scc_in_subgraph(&blocks, &repetition_arcs);
        for inner_scc in inner_sccs {
            if inner_scc.is_trivial {
                continue;
            }
            let inner_blocks: SmallVec<[BasicBlockId; 8]> =
                inner_scc.blocks.iter().copied().collect();
            let inner_id = self.alloc_node(inner_blocks, Some(node));
            self.children[node.0 as usize].push(inner_id);
            self.populate(inner_id, mapper);
        }
    }

    /// The collapse table of one nesting level: for every block, the
    /// component among `level` it belongs to (transitively -- an outer
    /// component's block set includes its children's). Used by the
    /// restructuring passes and the emitter to present each level's loops
    /// as single collapsed vertices; sharing one builder keeps the levels
    /// consistent across all consumers.
    pub fn collapse_table(
        &self,
        level: &[SccTreeNodeId],
        block_count: usize,
    ) -> Vec<Option<SccTreeNodeId>> {
        let mut table = vec![None; block_count];
        for &scc in level {
            for &block in &self.blocks[scc.0 as usize] {
                table[block.0 as usize] = Some(scc);
            }
        }
        table
    }

    /// Number of components in the tree, counting every nesting level.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Build a per-block lookup table: for every block in the function,
    /// records whether that block is an entry vertex of some component
    /// in the tree, and if so, which one. Region lowering uses this to
    /// decide when a block dispatches into a strongly connected
    /// component's theta node.
    ///
    /// When a block is the entry vertex of more than one component (an
    /// outer and an inner component sharing the same header, which is
    /// possible in pathological control flow), the table records the
    /// innermost. The region walker disambiguates by tracking which
    /// component it is currently lowering and skipping any entry that
    /// matches the current context.
    ///
    /// `block_count` is the total number of blocks in the function. The
    /// returned table has one entry per block, indexed by `BasicBlockId`.
    pub fn entry_block_to_node(&self, block_count: usize) -> Vec<Option<SccTreeNodeId>> {
        let mut table = vec![None; block_count];
        for (index, arcs) in self.arcs.iter().enumerate() {
            let node_id = SccTreeNodeId(index as u32);
            for &entry in &arcs.entry_blocks {
                // Inner components are populated after their parent, so
                // a later write here always represents an innermost
                // component for that entry vertex.
                table[entry.0 as usize] = Some(node_id);
            }
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::test_utils::init;
    use pretty_assertions::assert_eq;

    fn build(mapper: &BasicBlockMapper) -> SccTree {
        SccTree::build(mapper)
    }

    #[test]
    fn no_loops_produces_empty_tree() {
        // Straight-line CFG: 0 -> 1 -> 2. No cycles, no non-trivial
        // strongly connected components, no tree nodes.
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);

        let tree = build(&mapper);
        assert!(tree.is_empty());
        assert!(tree.roots.is_empty());
    }

    #[test]
    fn single_self_loop_is_one_root() {
        // 0 -> 1, 1 -> 1, 1 -> 2.
        let (mut mapper, bbs) = init(3);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);

        let tree = build(&mapper);
        assert_eq!(1, tree.len());
        assert_eq!(1, tree.roots.len());
        let root = tree.roots[0];
        assert_eq!(vec![bbs[1]], tree.blocks[root.0 as usize].to_vec());
        assert!(tree.children[root.0 as usize].is_empty());
        assert_eq!(None, tree.parent[root.0 as usize]);
    }

    #[test]
    fn two_disjoint_loops_are_two_roots() {
        // 0 -> 1, 1 -> 1 (loop A), 1 -> 2, 2 -> 3, 3 -> 3 (loop B), 3 -> 4.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);

        let tree = build(&mapper);
        assert_eq!(2, tree.len());
        assert_eq!(2, tree.roots.len());
        for &root in &tree.roots {
            assert_eq!(None, tree.parent[root.0 as usize]);
            assert!(tree.children[root.0 as usize].is_empty());
        }
    }

    #[test]
    fn nested_loops_06_shape_produces_parent_child() {
        // The 06_nested_loops shape: outer loop at 1 containing inner
        // self-loop at 2.
        //   0 -> 1 (outer header)
        //   1 -> 2 (inner header)
        //   2 -> 2 (inner self-loop)
        //   2 -> 3 (outer cond_block)
        //   3 -> 1 (outer back-edge)
        //   3 -> 4 (outer exit)
        //
        // Whole-function Tarjan merges {1, 2, 3} into one strongly
        // connected component. The SCC tree must distinguish the outer
        // component from the inner self-loop on 2 by removing the
        // outer's repetition arc 3 -> 1 and re-running Tarjan on the
        // outer's block set.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[1]);
        mapper.add_connection(bbs[3], bbs[4]);

        let tree = build(&mapper);
        assert_eq!(2, tree.len());
        assert_eq!(1, tree.roots.len());

        let outer = tree.roots[0];
        let mut outer_blocks = tree.blocks[outer.0 as usize].to_vec();
        outer_blocks.sort();
        assert_eq!(vec![bbs[1], bbs[2], bbs[3]], outer_blocks);
        assert_eq!(1, tree.children[outer.0 as usize].len());

        let inner = tree.children[outer.0 as usize][0];
        assert_eq!(vec![bbs[2]], tree.blocks[inner.0 as usize].to_vec());
        assert_eq!(Some(outer), tree.parent[inner.0 as usize]);
        assert!(tree.children[inner.0 as usize].is_empty());
    }

    #[test]
    fn triple_nested_produces_three_level_chain() {
        // Innermost on 3, middle on 2, outermost on 1. The CFG mirrors
        // what loop-rotate plus loop-simplify produce for three nested
        // do-while loops sharing the same control flow style.
        //   0 -> 1 (outer header)
        //   1 -> 2 (middle header)
        //   2 -> 3 (inner header)
        //   3 -> 3 (inner self-loop)
        //   3 -> 4 (middle latch source)
        //   4 -> 2 (middle back-edge)
        //   4 -> 5 (outer latch source)
        //   5 -> 1 (outer back-edge)
        //   5 -> 6 (outer exit)
        let (mut mapper, bbs) = init(7);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[3]);
        mapper.add_connection(bbs[3], bbs[4]);
        mapper.add_connection(bbs[4], bbs[2]);
        mapper.add_connection(bbs[4], bbs[5]);
        mapper.add_connection(bbs[5], bbs[1]);
        mapper.add_connection(bbs[5], bbs[6]);

        let tree = build(&mapper);
        assert_eq!(3, tree.len());
        assert_eq!(1, tree.roots.len());

        let outer = tree.roots[0];
        assert_eq!(1, tree.children[outer.0 as usize].len());
        let middle = tree.children[outer.0 as usize][0];
        assert_eq!(Some(outer), tree.parent[middle.0 as usize]);
        assert_eq!(1, tree.children[middle.0 as usize].len());
        let inner = tree.children[middle.0 as usize][0];
        assert_eq!(Some(middle), tree.parent[inner.0 as usize]);
        assert!(tree.children[inner.0 as usize].is_empty());

        // Block sets are increasingly small as we descend.
        assert!(tree.blocks[outer.0 as usize].len() > tree.blocks[middle.0 as usize].len());
        assert!(tree.blocks[middle.0 as usize].len() > tree.blocks[inner.0 as usize].len());
    }

    #[test]
    fn entry_block_table_maps_each_header() {
        // Nested loops: outer header is 1, inner header is 2.
        let (mut mapper, bbs) = init(5);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[2]);
        mapper.add_connection(bbs[2], bbs[3]);
        mapper.add_connection(bbs[3], bbs[1]);
        mapper.add_connection(bbs[3], bbs[4]);

        let tree = build(&mapper);
        let table = tree.entry_block_to_node(mapper.blocks.len());

        // Block 0 is outside any loop.
        assert_eq!(None, table[bbs[0].0 as usize]);
        // Block 1 is the outer entry vertex.
        let outer_id = table[bbs[1].0 as usize].expect("outer entry should map");
        // Block 2 is the inner entry vertex.
        let inner_id = table[bbs[2].0 as usize].expect("inner entry should map");
        // Block 3 is in the outer body but not an entry vertex.
        assert_eq!(None, table[bbs[3].0 as usize]);
        // Block 4 is outside any loop.
        assert_eq!(None, table[bbs[4].0 as usize]);

        // The two IDs should differ: outer wraps inner.
        assert_ne!(outer_id, inner_id);
        assert_eq!(None, tree.parent[outer_id.0 as usize]);
        assert_eq!(Some(outer_id), tree.parent[inner_id.0 as usize]);
    }

    #[test]
    fn loop_arcs_attached_to_each_node() {
        // 0 -> 1 -> 2 -> 1, 1 -> 3 exit.
        let (mut mapper, bbs) = init(4);
        mapper.add_connection(bbs[0], bbs[1]);
        mapper.add_connection(bbs[1], bbs[2]);
        mapper.add_connection(bbs[2], bbs[1]);
        mapper.add_connection(bbs[1], bbs[3]);

        let tree = build(&mapper);
        assert_eq!(1, tree.len());
        let arcs = &tree.arcs[tree.roots[0].0 as usize];
        assert_eq!(vec![bbs[1]], arcs.entry_blocks.to_vec());
        assert_eq!(vec![(bbs[0], bbs[1])], arcs.entry_arcs.to_vec());
        assert_eq!(vec![bbs[3]], arcs.exit_blocks.to_vec());
        assert_eq!(vec![(bbs[1], bbs[3])], arcs.exit_arcs.to_vec());
        assert_eq!(vec![(bbs[2], bbs[1])], arcs.repetition_arcs.to_vec());
    }
}
