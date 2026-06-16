use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockInOuts, BasicBlockMapper};
use smallvec::SmallVec;
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub struct SccAnalysis {
    /// The components in reverse-topological order (reverse for forward order).
    pub sccs: Vec<Scc>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Scc {
    pub blocks: SmallVec<[BasicBlockId; 4]>,
    pub is_trivial: bool,
}
/// Tarjan's strongly-connected-components core, shared by the whole-function
/// pass and the subgraph pass. `mask` (when `Some`) restricts the analysis to
/// blocks marked `true`; `exclude` lists arcs treated as absent (the paper's
/// "remove repetition arcs" for nested-loop discovery). The whole-function pass
/// is just `mask = None, exclude = &[]`. Components are produced in reverse
/// topological order; an id is a component's index in `sccs`.
struct Tarjan<'a> {
    blocks: &'a [BasicBlockInOuts],
    mask: Option<&'a [bool]>,
    exclude: &'a [(BasicBlockId, BasicBlockId)],
    index: i32,
    indices: Vec<i32>,
    low_links: Vec<i32>,
    on_stack: Vec<bool>,
    stack: VecDeque<BasicBlockId>,
    sccs: Vec<Scc>,
}

impl<'a> Tarjan<'a> {
    fn new(
        blocks: &'a [BasicBlockInOuts],
        mask: Option<&'a [bool]>,
        exclude: &'a [(BasicBlockId, BasicBlockId)],
    ) -> Self {
        let n = blocks.len();
        Self {
            blocks,
            mask,
            exclude,
            index: 0,
            indices: vec![-1; n],
            low_links: vec![0; n],
            on_stack: vec![false; n],
            stack: VecDeque::new(),
            sccs: Vec::new(),
        }
    }

    /// Whether the arc `from -> to` is part of the analysed (sub)graph: its
    /// target must be in the mask (if any), and the arc must not be excluded.
    #[inline]
    fn edge_live(&self, from: BasicBlockId, to: BasicBlockId) -> bool {
        self.mask.map_or(true, |m| m[to.0 as usize])
            && (self.exclude.is_empty() || !self.exclude.contains(&(from, to)))
    }

    /// Run the SCC search from every not-yet-visited root in `roots`.
    fn run(&mut self, roots: impl Iterator<Item = BasicBlockId>) {
        for root in roots {
            if self.indices[root.0 as usize] == -1 {
                self.strong_connect(root);
            }
        }
    }

    fn strong_connect(&mut self, id: BasicBlockId) {
        let i = id.0 as usize;
        self.indices[i] = self.index;
        self.low_links[i] = self.index;
        self.index += 1;
        self.stack.push_back(id);
        self.on_stack[i] = true;

        // `blocks` is a shared-reference field, so copying it out lets the
        // recursive `&mut self` call coexist with iterating this block's edges.
        let blocks = self.blocks;
        for &edge in &blocks[i].outputs {
            if !self.edge_live(id, edge) {
                continue;
            }
            let e = edge.0 as usize;
            if self.indices[e] == -1 {
                self.strong_connect(edge);
                self.low_links[i] = self.low_links[i].min(self.low_links[e]);
            } else if self.on_stack[e] {
                self.low_links[i] = self.low_links[i].min(self.indices[e]);
            }
        }

        if self.low_links[i] == self.indices[i] {
            let mut scc = Scc {
                blocks: SmallVec::new(),
                is_trivial: false,
            };
            while let Some(node) = self.stack.pop_back() {
                self.on_stack[node.0 as usize] = false;
                scc.blocks.push(node);
                if node == id {
                    break;
                }
            }
            // Trivial = a single block with no self-edge surviving the filter.
            // Multi-block components are always cyclic; a surviving self-edge
            // makes a singleton a (real, if trivial-looking) loop.
            let self_loop = blocks[i].outputs.contains(&id) && self.edge_live(id, id);
            scc.is_trivial = scc.blocks.len() == 1 && !self_loop;
            if !scc.blocks.is_empty() {
                self.sccs.push(scc);
            }
        }
    }
}

impl BasicBlockMapper {
    pub fn get_strongly_connected_components(&self) -> SccAnalysis {
        let mut tarjan = Tarjan::new(&self.blocks, None, &[]);
        tarjan.run((0..self.blocks.len() as u32).map(BasicBlockId));
        SccAnalysis { sccs: tarjan.sccs }
    }

    /// Compute strongly connected components over a subgraph induced by
    /// `blocks`, treating arcs in `exclude_arcs` as absent. Returns the
    /// strongly connected components of the resulting directed graph in
    /// reverse topological order. Both trivial single-block components
    /// and non-trivial multi-block (or self-looping single-block)
    /// components are included; callers filter for the kind they want.
    ///
    /// This is the analysis backbone for nested loop discovery in the
    /// paper-faithful construction (see construction_plan.md, section
    /// 4.9). After the whole-function strongly connected components have
    /// been identified, recursing into each non-trivial component's
    /// block set with that component's repetition arcs excluded reveals
    /// the inner loops without ever re-running Tarjan over the full
    /// function.
    ///
    /// Complexity: O(|blocks| + |edges-in-subgraph|) per call, plus an
    /// O(N) zero-fill of three working arrays sized for the whole
    /// function. Memory is owned by the call; nothing persists.
    pub fn scc_in_subgraph(
        &self,
        blocks: &[BasicBlockId],
        exclude_arcs: &[(BasicBlockId, BasicBlockId)],
    ) -> Vec<Scc> {
        // Restrict Tarjan to `blocks` via a membership mask, with `exclude_arcs`
        // removed. Roots are the subgraph's blocks; blocks outside it are never
        // visited (their indices stay -1) and can't be reached through edges
        // the mask filters out.
        let mut in_subgraph = vec![false; self.blocks.len()];
        for b in blocks {
            in_subgraph[b.0 as usize] = true;
        }
        let mut tarjan = Tarjan::new(&self.blocks, Some(&in_subgraph), exclude_arcs);
        tarjan.run(blocks.iter().copied());
        tarjan.sccs
    }
}

#[cfg(test)]
mod tests {
    use llvm_ir::Name;
    use pretty_assertions::assert_eq;

    use super::SccAnalysis;
    use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockMapper};

    fn init(n: usize) -> (BasicBlockMapper, Vec<BasicBlockId>) {
        let mut basic_blocks = BasicBlockMapper::new(n);
        let bbs = (0..n)
            .map(|i| basic_blocks.intern(&Name::Number(i)))
            .collect::<Vec<_>>();
        (basic_blocks, bbs)
    }

    /// Pull the SCC block-lists out of an analysis result and normalize ordering
    /// so tests can assert on a deterministic shape.
    fn collect_sorted(analysis: SccAnalysis) -> Vec<Vec<BasicBlockId>> {
        let mut sccs: Vec<Vec<BasicBlockId>> = analysis
            .sccs
            .iter()
            .map(|scc| scc.blocks.iter().copied().collect())
            .collect();
        for grp in sccs.iter_mut() {
            grp.sort();
        }
        sccs.sort();
        sccs
    }

    #[test]
    fn basic_scc() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[2]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0], bbs[1],], vec![bbs[2], bbs[3], bbs[4],]],
            sccs
        );
    }

    #[test]
    fn single_node_no_edges() {
        let (basic_blocks, bbs) = init(1);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0]]], sccs);
    }

    #[test]
    fn single_node_with_self_loop() {
        let (mut basic_blocks, bbs) = init(1);

        basic_blocks.add_connection(bbs[0], bbs[0]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0]]], sccs);
    }

    #[test]
    fn two_nodes_single_direction() {
        let (mut basic_blocks, bbs) = init(2);

        basic_blocks.add_connection(bbs[0], bbs[1]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0]], vec![bbs[1]]], sccs);
    }

    #[test]
    fn two_nodes_bidirectional() {
        let (mut basic_blocks, bbs) = init(2);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1]]], sccs);
    }

    #[test]
    fn linear_chain_five_nodes() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0]],
                vec![bbs[1]],
                vec![bbs[2]],
                vec![bbs[3]],
                vec![bbs[4]],
            ],
            sccs
        );
    }

    #[test]
    fn simple_cycle_three_nodes() {
        let (mut basic_blocks, bbs) = init(3);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2]]], sccs);
    }

    #[test]
    fn large_cycle_seven_nodes() {
        let (mut basic_blocks, bbs) = init(7);

        for i in 0..7 {
            basic_blocks.add_connection(bbs[i], bbs[(i + 1) % 7]);
        }

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0], bbs[1], bbs[2], bbs[3], bbs[4], bbs[5], bbs[6],]],
            sccs
        );
    }

    #[test]
    fn dag_diamond_no_cycles() {
        let (mut basic_blocks, bbs) = init(4);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[0], bbs[2]);
        basic_blocks.add_connection(bbs[1], bbs[3]);
        basic_blocks.add_connection(bbs[2], bbs[3]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0]], vec![bbs[1]], vec![bbs[2]], vec![bbs[3]]],
            sccs
        );
    }

    #[test]
    fn complete_graph_three_nodes() {
        let (mut basic_blocks, bbs) = init(3);

        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    basic_blocks.add_connection(bbs[i], bbs[j]);
                }
            }
        }

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2]]], sccs);
    }

    #[test]
    fn star_graph_outbound() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[0], bbs[2]);
        basic_blocks.add_connection(bbs[0], bbs[3]);
        basic_blocks.add_connection(bbs[0], bbs[4]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0]],
                vec![bbs[1]],
                vec![bbs[2]],
                vec![bbs[3]],
                vec![bbs[4]],
            ],
            sccs
        );
    }

    #[test]
    fn two_cycles_with_bridge() {
        let (mut basic_blocks, bbs) = init(6);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);

        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[5]);
        basic_blocks.add_connection(bbs[5], bbs[3]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0], bbs[1], bbs[2]], vec![bbs[3], bbs[4], bbs[5]],],
            sccs
        );
    }

    #[test]
    fn cycle_with_tail_entering() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[2]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0]], vec![bbs[1]], vec![bbs[2], bbs[3], bbs[4]],],
            sccs
        );
    }

    #[test]
    fn cycle_with_tail_exiting() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0], bbs[1], bbs[2]], vec![bbs[3]], vec![bbs[4]],],
            sccs
        );
    }

    #[test]
    fn figure_eight_shared_node() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[2]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2], bbs[3], bbs[4]]], sccs);
    }

    #[test]
    fn self_loop_inside_cycle() {
        let (mut basic_blocks, bbs) = init(3);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2]]], sccs);
    }

    #[test]
    fn tarjan_classic_example() {
        // Classic example from Tarjan's paper:
        // 0 -> 1, 1 -> 2, 2 -> 0 (scc: {0,1,2})
        // 3 -> 1, 3 -> 2, 3 -> 4
        // 4 -> 3, 4 -> 5 (scc: {3,4})
        // 5 -> 2, 5 -> 6
        // 6 -> 5 (scc: {5,6})
        // 7 -> 4, 7 -> 6, 7 -> 7 (scc: {7})
        let (mut basic_blocks, bbs) = init(8);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        basic_blocks.add_connection(bbs[3], bbs[1]);
        basic_blocks.add_connection(bbs[3], bbs[2]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        basic_blocks.add_connection(bbs[4], bbs[3]);
        basic_blocks.add_connection(bbs[4], bbs[5]);

        basic_blocks.add_connection(bbs[5], bbs[2]);
        basic_blocks.add_connection(bbs[5], bbs[6]);

        basic_blocks.add_connection(bbs[6], bbs[5]);

        basic_blocks.add_connection(bbs[7], bbs[4]);
        basic_blocks.add_connection(bbs[7], bbs[6]);
        basic_blocks.add_connection(bbs[7], bbs[7]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0], bbs[1], bbs[2]],
                vec![bbs[3], bbs[4]],
                vec![bbs[5], bbs[6]],
                vec![bbs[7]],
            ],
            sccs
        );
    }

    #[test]
    fn back_edge_from_deep_node() {
        let (mut basic_blocks, bbs) = init(6);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[5]);
        basic_blocks.add_connection(bbs[5], bbs[0]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0], bbs[1], bbs[2], bbs[3], bbs[4], bbs[5],]],
            sccs
        );
    }

    #[test]
    fn nested_cycle_inner_and_outer() {
        let (mut basic_blocks, bbs) = init(4);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[1]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[0]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2], bbs[3]]], sccs);
    }

    #[test]
    fn branching_dag_many_paths() {
        let (mut basic_blocks, bbs) = init(6);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[0], bbs[2]);
        basic_blocks.add_connection(bbs[1], bbs[3]);
        basic_blocks.add_connection(bbs[1], bbs[4]);
        basic_blocks.add_connection(bbs[2], bbs[4]);
        basic_blocks.add_connection(bbs[2], bbs[5]);
        basic_blocks.add_connection(bbs[3], bbs[5]);
        basic_blocks.add_connection(bbs[4], bbs[5]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0]],
                vec![bbs[1]],
                vec![bbs[2]],
                vec![bbs[3]],
                vec![bbs[4]],
                vec![bbs[5]],
            ],
            sccs
        );
    }

    #[test]
    fn two_sccs_chained_by_one_edge() {
        let (mut basic_blocks, bbs) = init(4);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        basic_blocks.add_connection(bbs[1], bbs[2]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[2]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1]], vec![bbs[2], bbs[3]]], sccs);
    }

    #[test]
    fn cycle_with_two_branching_tails() {
        let (mut basic_blocks, bbs) = init(7);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        basic_blocks.add_connection(bbs[2], bbs[5]);
        basic_blocks.add_connection(bbs[5], bbs[6]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0], bbs[1], bbs[2]],
                vec![bbs[3]],
                vec![bbs[4]],
                vec![bbs[5]],
                vec![bbs[6]],
            ],
            sccs
        );
    }

    #[test]
    fn two_cycles_converging_to_shared_scc() {
        let (mut basic_blocks, bbs) = init(7);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[2]);

        basic_blocks.add_connection(bbs[1], bbs[4]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        basic_blocks.add_connection(bbs[4], bbs[5]);
        basic_blocks.add_connection(bbs[5], bbs[6]);
        basic_blocks.add_connection(bbs[6], bbs[4]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0], bbs[1]],
                vec![bbs[2], bbs[3]],
                vec![bbs[4], bbs[5], bbs[6]],
            ],
            sccs
        );
    }

    #[test]
    fn complete_bidirectional_four_nodes() {
        let (mut basic_blocks, bbs) = init(4);

        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    basic_blocks.add_connection(bbs[i], bbs[j]);
                }
            }
        }

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2], bbs[3]]], sccs);
    }

    #[test]
    fn cross_edges_between_branches() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[0], bbs[2]);
        basic_blocks.add_connection(bbs[1], bbs[3]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0]],
                vec![bbs[1]],
                vec![bbs[2]],
                vec![bbs[3]],
                vec![bbs[4]],
            ],
            sccs
        );
    }

    #[test]
    fn long_chain_of_sccs() {
        let (mut basic_blocks, bbs) = init(8);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[2]);

        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[5]);
        basic_blocks.add_connection(bbs[5], bbs[4]);

        basic_blocks.add_connection(bbs[5], bbs[6]);
        basic_blocks.add_connection(bbs[6], bbs[7]);
        basic_blocks.add_connection(bbs[7], bbs[6]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![
                vec![bbs[0], bbs[1]],
                vec![bbs[2], bbs[3]],
                vec![bbs[4], bbs[5]],
                vec![bbs[6], bbs[7]],
            ],
            sccs
        );
    }

    #[test]
    fn reverse_order_edges() {
        let (mut basic_blocks, bbs) = init(4);

        basic_blocks.add_connection(bbs[3], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);
        basic_blocks.add_connection(bbs[0], bbs[3]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2], bbs[3]]], sccs);
    }

    #[test]
    fn cycle_entered_from_middle_node() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[1]);

        basic_blocks.add_connection(bbs[0], bbs[2]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0]], vec![bbs[1], bbs[2], bbs[3]], vec![bbs[4]],],
            sccs
        );
    }

    #[test]
    fn two_scc_with_back_and_forward_connections() {
        let (mut basic_blocks, bbs) = init(6);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[5]);
        basic_blocks.add_connection(bbs[5], bbs[3]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[0], bbs[5]);

        let sccs = collect_sorted(basic_blocks.get_strongly_connected_components());

        assert_eq!(
            vec![vec![bbs[0], bbs[1], bbs[2]], vec![bbs[3], bbs[4], bbs[5]],],
            sccs
        );
    }

    /// Look up an SCC by any block it contains.
    fn scc_for(analysis: &SccAnalysis, block: BasicBlockId) -> &super::Scc {
        analysis
            .sccs
            .iter()
            .find(|s| s.blocks.contains(&block))
            .expect("block should be in some SCC")
    }

    #[test]
    fn is_trivial_singleton_no_edges() {
        let (basic_blocks, bbs) = init(1);
        let analysis = basic_blocks.get_strongly_connected_components();
        assert!(scc_for(&analysis, bbs[0]).is_trivial);
    }

    #[test]
    fn is_trivial_singleton_with_self_loop() {
        let (mut basic_blocks, bbs) = init(1);
        basic_blocks.add_connection(bbs[0], bbs[0]);
        let analysis = basic_blocks.get_strongly_connected_components();
        // single block + self-edge = a (degenerate) loop, NOT trivial
        assert!(!scc_for(&analysis, bbs[0]).is_trivial);
    }

    #[test]
    fn is_trivial_multi_block_cycle() {
        let (mut basic_blocks, bbs) = init(3);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);
        let analysis = basic_blocks.get_strongly_connected_components();
        assert!(!scc_for(&analysis, bbs[0]).is_trivial);
    }

    /// Normalise scc_in_subgraph output for assertions: sort blocks
    /// inside each SCC, then sort SCCs by their first block.
    fn collect_subgraph_sorted(sccs: Vec<super::Scc>) -> Vec<Vec<BasicBlockId>> {
        let mut out: Vec<Vec<BasicBlockId>> = sccs
            .into_iter()
            .map(|scc| {
                let mut bs: Vec<BasicBlockId> = scc.blocks.into_iter().collect();
                bs.sort();
                bs
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn subgraph_empty_blocks_returns_no_sccs() {
        let (mut basic_blocks, bbs) = init(3);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        let sccs = basic_blocks.scc_in_subgraph(&[], &[]);
        assert!(sccs.is_empty());
    }

    #[test]
    fn subgraph_excludes_blocks_outside_set() {
        // Whole-function SCCs: {0,1} and {2,3}. Restricting to {0,1}
        // returns only the first.
        let (mut basic_blocks, bbs) = init(4);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[2]);

        let sccs = collect_subgraph_sorted(basic_blocks.scc_in_subgraph(&[bbs[0], bbs[1]], &[]));
        assert_eq!(vec![vec![bbs[0], bbs[1]]], sccs);
    }

    #[test]
    fn subgraph_excluded_arc_breaks_cycle() {
        // 0 <-> 1 forms an SCC. Excluding the back-edge 1 -> 0 breaks
        // the cycle; both blocks become trivial singletons.
        let (mut basic_blocks, bbs) = init(2);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        let sccs = collect_subgraph_sorted(
            basic_blocks.scc_in_subgraph(&[bbs[0], bbs[1]], &[(bbs[1], bbs[0])]),
        );
        assert_eq!(vec![vec![bbs[0]], vec![bbs[1]]], sccs);
    }

    #[test]
    fn subgraph_excluded_self_loop_makes_singleton_trivial() {
        // Block with self-loop. Excluding the self-loop should make the
        // singleton trivial.
        let (mut basic_blocks, bbs) = init(1);
        basic_blocks.add_connection(bbs[0], bbs[0]);

        let raw = basic_blocks.scc_in_subgraph(&[bbs[0]], &[(bbs[0], bbs[0])]);
        assert_eq!(1, raw.len());
        assert!(raw[0].is_trivial);
    }

    #[test]
    fn subgraph_preserved_self_loop_keeps_singleton_non_trivial() {
        let (mut basic_blocks, bbs) = init(1);
        basic_blocks.add_connection(bbs[0], bbs[0]);

        let raw = basic_blocks.scc_in_subgraph(&[bbs[0]], &[]);
        assert_eq!(1, raw.len());
        assert!(!raw[0].is_trivial);
    }

    #[test]
    fn subgraph_reveals_nested_inner_loop_after_outer_back_edge_removed() {
        // Outer loop 0 -> 1 -> 2 -> 1 (inner self-cycle on the path) -> 3 -> 0.
        // Whole-function Tarjan merges everything into {0,1,2,3}.
        // Restricting to that block set and excluding the outer
        // repetition arc 3 -> 0 reveals the inner cycle {1,2} as a
        // separate SCC.
        let (mut basic_blocks, bbs) = init(4);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[1]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[0]);

        let inner_blocks = [bbs[0], bbs[1], bbs[2], bbs[3]];
        let exclude = [(bbs[3], bbs[0])];
        let sccs = collect_subgraph_sorted(basic_blocks.scc_in_subgraph(&inner_blocks, &exclude));
        // After excluding the back-edge, {1,2} is still a cycle; {0}
        // and {3} are reachable but no longer participate in a cycle.
        assert_eq!(vec![vec![bbs[0]], vec![bbs[1], bbs[2]], vec![bbs[3]]], sccs);
    }

    #[test]
    fn subgraph_ignores_edges_leaving_the_subgraph() {
        // Block 0 has an outgoing edge to block 2 (not in subgraph).
        // The subgraph {0, 1} should not see that edge at all.
        let (mut basic_blocks, bbs) = init(3);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);
        basic_blocks.add_connection(bbs[0], bbs[2]);

        let sccs = collect_subgraph_sorted(basic_blocks.scc_in_subgraph(&[bbs[0], bbs[1]], &[]));
        assert_eq!(vec![vec![bbs[0], bbs[1]]], sccs);
    }

    #[test]
    fn is_trivial_mixed_graph() {
        // 0 (singleton, no self-loop) -> 1 -> 2 -> 1 (loop) -> 3 (singleton, no self-loop)
        let (mut basic_blocks, bbs) = init(4);
        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[3]);
        let analysis = basic_blocks.get_strongly_connected_components();

        assert!(scc_for(&analysis, bbs[0]).is_trivial);
        assert!(!scc_for(&analysis, bbs[1]).is_trivial); // {1,2} is the loop
        assert!(!scc_for(&analysis, bbs[2]).is_trivial);
        assert!(scc_for(&analysis, bbs[3]).is_trivial);
    }
}
