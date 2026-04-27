use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockInOuts, BasicBlockMapper};
use smallvec::SmallVec;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SccId(u32);

#[derive(Clone, Debug, PartialEq)]
pub struct SccAnalysis {
    /// reverse topo order; reverse for forward
    pub sccs: Vec<Scc>,
    pub block_to_scc: Vec<SccId>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Scc {
    pub blocks: SmallVec<[BasicBlockId; 4]>,
    pub is_trivial: bool,
}
impl SccAnalysis {
    fn add_scc(&mut self) -> Scc {
        let scc = Scc::default();
        self.sccs.push(scc);
        scc
    }
}

impl BasicBlockMapper {
    pub fn get_strongly_connected_components(&self) -> SccAnalysis {
        let mut scc_analysis = SccAnalysis {
            sccs: vec![],
            block_to_scc: vec![],
        };
        let mut stack = VecDeque::new();
        let mut index = 0;
        let mut indicies = vec![-1; self.blocks.len()];
        let mut low_links = vec![0; self.blocks.len()];
        let mut on_stack = vec![false; self.blocks.len()];

        #[inline]
        fn strong_connect(
            id: BasicBlockId,
            blocks: &[BasicBlockInOuts],
            scc_analysis: &mut SccAnalysis,
            stack: &mut VecDeque<BasicBlockId>,
            index: &mut i32,
            indicies: &mut Vec<i32>,
            low_links: &mut Vec<i32>,
            on_stack: &mut Vec<bool>,
        ) {
            // add current id to indicies, low_link and stack
            indicies[id.0 as usize] = *index;
            low_links[id.0 as usize] = *index;
            *index += 1;
            stack.push_back(id);
            on_stack[id.0 as usize] = true;

            let block = &blocks[id.0 as usize];
            for edge in &block.outputs {
                if indicies[edge.0 as usize] == -1 {
                    // edge has not yet been visited, recusively do it.
                    strong_connect(
                        *edge,
                        blocks,
                        scc_analysis,
                        stack,
                        index,
                        indicies,
                        low_links,
                        on_stack,
                    );
                    low_links[id.0 as usize] =
                        low_links[id.0 as usize].min(low_links[edge.0 as usize]);
                } else if on_stack[edge.0 as usize] {
                    // edge is on the stack and therefor is the current scc
                    low_links[id.0 as usize] =
                        low_links[id.0 as usize].min(indicies[edge.0 as usize]);
                }
            }

            if low_links[id.0 as usize] == indicies[id.0 as usize] {
                let mut scc = Scc {
                    blocks: SmallVec::new(),
                    is_trivial: true,
                };
                // TODO: compute the is_trivial
                while let Some(node) = stack.pop_back() {
                    on_stack[node.0 as usize] = false;
                    let scc_id = SccId(scc.blocks.len() as u32);
                    scc.blocks.push(node);
                    scc_analysis.block_to_scc[node.0 as usize] = scc_id;
                    if node == id {
                        break;
                    }
                }
                if scc.blocks.len() > 0 {
                    scc_analysis.sccs.push(scc);
                }
            }
        }

        for i in 0..self.blocks.len() {
            if indicies[i] == -1 {
                strong_connect(
                    BasicBlockId(i as u32),
                    &self.blocks,
                    &mut scc_analysis,
                    &mut stack,
                    &mut index,
                    &mut indicies,
                    &mut low_links,
                    &mut on_stack,
                );
            }
        }

        scc_analysis
    }
}

#[cfg(test)]
mod tests {
    use llvm_ir::Name;
    use pretty_assertions::assert_eq;

    use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockMapper};

    fn init(n: usize) -> (BasicBlockMapper, Vec<BasicBlockId>) {
        let mut basic_blocks = BasicBlockMapper::new(n);
        let bbs = (0..n)
            .map(|i| basic_blocks.intern(&Name::Number(i)))
            .collect::<Vec<_>>();
        (basic_blocks, bbs)
    }

    fn sort(sccs: &mut Vec<Vec<BasicBlockId>>) {
        for grp in sccs.iter_mut() {
            grp.sort();
        }
        sccs.sort();
    }

    #[test]
    fn basic_scc() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);
        basic_blocks.add_connection(bbs[4], bbs[2]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(
            vec![vec![bbs[0], bbs[1],], vec![bbs[2], bbs[3], bbs[4],]],
            sccs
        );
    }

    #[test]
    fn single_node_no_edges() {
        let (basic_blocks, bbs) = init(1);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0]]], sccs);
    }

    #[test]
    fn single_node_with_self_loop() {
        let (mut basic_blocks, bbs) = init(1);

        basic_blocks.add_connection(bbs[0], bbs[0]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0]]], sccs);
    }

    #[test]
    fn two_nodes_single_direction() {
        let (mut basic_blocks, bbs) = init(2);

        basic_blocks.add_connection(bbs[0], bbs[1]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0]], vec![bbs[1]]], sccs);
    }

    #[test]
    fn two_nodes_bidirectional() {
        let (mut basic_blocks, bbs) = init(2);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[0]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0], bbs[1]]], sccs);
    }

    #[test]
    fn linear_chain_five_nodes() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[3]);
        basic_blocks.add_connection(bbs[3], bbs[4]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2]]], sccs);
    }

    #[test]
    fn large_cycle_seven_nodes() {
        let (mut basic_blocks, bbs) = init(7);

        for i in 0..7 {
            basic_blocks.add_connection(bbs[i], bbs[(i + 1) % 7]);
        }

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2]]], sccs);
    }

    #[test]
    fn star_graph_outbound() {
        let (mut basic_blocks, bbs) = init(5);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[0], bbs[2]);
        basic_blocks.add_connection(bbs[0], bbs[3]);
        basic_blocks.add_connection(bbs[0], bbs[4]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(vec![vec![bbs[0], bbs[1], bbs[2], bbs[3], bbs[4]]], sccs);
    }

    #[test]
    fn self_loop_inside_cycle() {
        let (mut basic_blocks, bbs) = init(3);

        basic_blocks.add_connection(bbs[0], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[1]);
        basic_blocks.add_connection(bbs[1], bbs[2]);
        basic_blocks.add_connection(bbs[2], bbs[0]);

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

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

        let mut sccs = basic_blocks.get_strongly_connected_components();
        sort(&mut sccs);

        assert_eq!(
            vec![vec![bbs[0], bbs[1], bbs[2]], vec![bbs[3], bbs[4], bbs[5]],],
            sccs
        );
    }
}
