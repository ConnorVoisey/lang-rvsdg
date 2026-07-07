//! **The restructuring oracle** (test-only): the machine-checkable proof
//! that the overlay follows the paper.
//!
//! The paper's Corollary 5.7 states that short-circuiting the restructured
//! CFG recovers the original exactly:
//!
//!     SHORTCIRCUITCFG(RESTRUCTURECFG(C)) = C
//!
//! where SHORTCIRCUITCFG (its section 5.2) is a restricted constant
//! propagation over the auxiliary variables: an assignment of a constant
//! selector followed by a branch on that selector resolves to a direct arc,
//! drained assignment vertices and null vertices are bypassed, and
//! unreachable leftovers are pruned. Since restructuring only ever inserts
//! constant selector assignments and branches on them, everything it
//! inserted dissolves and the original arcs reappear -- if and only if the
//! restructuring was faithful.
//!
//! This module expands the overlay into the explicit statement CFG it
//! denotes (undoing the virtual collapse: a redirect into a collapsed loop
//! becomes an arc to the loop's real entry point, and the loop tail's real
//! repetition and exit arcs are materialized), runs the short-circuit, and
//! compares the result against the original closed CFG per block, in arc
//! order. It runs in tests only; production never materializes any CFG.

#![cfg(test)]

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    control_flow::overlay::{ArcId, AuxAssign, AuxVar, AuxVertexKind, Overlay, Vertex},
    scc::{SccTree, SccTreeNodeId},
};

/// Node ids: `0..block_count` are the original blocks (including the
/// synthetic exit), `block_count..block_count + aux_count` the inserted
/// vertices, and everything after is an assignment cluster materialized
/// from a rewritten arc.
type Node = usize;

struct Expansion {
    block_count: usize,
    /// Ordered successors (arc index = alternative).
    succs: Vec<Vec<Node>>,
    /// For a demux or loop-tail node: the selector variable it branches on.
    branch: Vec<Option<AuxVar>>,
    /// Aligned with `succs` for branch nodes: the selector value that picks
    /// each slot.
    values: Vec<Vec<u32>>,
    /// The constant assignments a cluster or promoted node performs, in
    /// evaluation order.
    assigns: Vec<Vec<AuxAssign>>,
}

impl Expansion {
    fn add_node(&mut self) -> Node {
        self.succs.push(Vec::new());
        self.branch.push(None);
        self.values.push(Vec::new());
        self.assigns.push(Vec::new());
        self.succs.len() - 1
    }
}

/// Materialize the statement CFG the overlay denotes.
fn expand(mapper: &BasicBlockMapper, tree: &SccTree, overlay: &Overlay) -> Expansion {
    let block_count = mapper.blocks.len();
    let aux_count = overlay.aux_vertices.len();
    let mut expansion = Expansion {
        block_count,
        succs: vec![Vec::new(); block_count + aux_count],
        branch: vec![None; block_count + aux_count],
        values: vec![Vec::new(); block_count + aux_count],
        assigns: vec![Vec::new(); block_count + aux_count],
    };

    // A redirect into a collapsed loop means "enter the loop": the real
    // destination is its entry demux, or its single entry vertex.
    let resolve = |target: Vertex| -> Node {
        match target {
            Vertex::Block(b) => b.0 as usize,
            Vertex::Aux(a) => block_count + a.0 as usize,
            Vertex::Loop(scc) => {
                let record = overlay.loops[scc.0 as usize]
                    .as_ref()
                    .expect("collapsed loop without a record");
                match record.entry_demux {
                    Some(demux) => block_count + demux.0 as usize,
                    None => record.entries[0].0 as usize,
                }
            }
        }
    };

    // One arc, possibly through a materialized assignment cluster.
    let add_arc = |expansion: &mut Expansion, from: Node, arc: ArcId, raw_target: Vertex| {
        let (assignments, target) = match overlay.rewrite_of(arc) {
            Some(rewrite) => (rewrite.assignments.as_slice(), rewrite.redirect),
            None => (&[][..], raw_target),
        };
        let dest = resolve(target);
        if assignments.is_empty() {
            expansion.succs[from].push(dest);
        } else {
            let cluster = expansion.add_node();
            expansion.assigns[cluster] = assignments.to_vec();
            expansion.succs[cluster].push(dest);
            expansion.succs[from].push(cluster);
        }
    };

    // Original block arcs (a diverging block gets its synthetic arc to the
    // exit: the closed-CFG convention the whole pipeline works over).
    for index in 0..block_count {
        let block = BasicBlockId(index as u32);
        if overlay.is_diverging(block) {
            add_arc(
                &mut expansion,
                index,
                ArcId {
                    source: Vertex::Block(block),
                    index: 0,
                },
                Vertex::Block(overlay.exit_block),
            );
            continue;
        }
        for (arc_index, &target) in mapper.outputs(block).iter().enumerate() {
            add_arc(
                &mut expansion,
                index,
                ArcId {
                    source: Vertex::Block(block),
                    index: arc_index as u32,
                },
                Vertex::Block(target),
            );
        }
    }

    // Which loop each tail belongs to (the variant carries no field).
    let mut tail_loop: FxHashMap<usize, SccTreeNodeId> = FxHashMap::default();
    for (index, record) in overlay.loops.iter().enumerate() {
        if let Some(record) = record
            && let Some(tail) = record.tail
        {
            tail_loop.insert(tail.0 as usize, SccTreeNodeId(index as u32));
        }
    }
    let _ = tree;

    // Inserted vertices.
    for (aux_index, vertex) in overlay.aux_vertices.iter().enumerate() {
        let node = block_count + aux_index;
        match &vertex.kind {
            AuxVertexKind::LoopEntryDemux { scc } | AuxVertexKind::LoopExitDemux { scc } => {
                expansion.branch[node] = Some(AuxVar::LoopVertexSelector(*scc));
                for (k, fan_out) in vertex.fan_out.iter().enumerate() {
                    expansion.values[node].push(k as u32);
                    add_arc(
                        &mut expansion,
                        node,
                        ArcId {
                            source: Vertex::Aux(
                                crate::llvm_parser::control_flow::overlay::AuxVertexId(
                                    aux_index as u32,
                                ),
                            ),
                            index: k as u32,
                        },
                        fan_out.target,
                    );
                }
            }
            AuxVertexKind::BranchDemux => {
                expansion.branch[node] = Some(AuxVar::ContinuationSelector(
                    crate::llvm_parser::control_flow::overlay::AuxVertexId(aux_index as u32),
                ));
                for (k, fan_out) in vertex.fan_out.iter().enumerate() {
                    expansion.values[node].push(k as u32);
                    add_arc(
                        &mut expansion,
                        node,
                        ArcId {
                            source: Vertex::Aux(
                                crate::llvm_parser::control_flow::overlay::AuxVertexId(
                                    aux_index as u32,
                                ),
                            ),
                            index: k as u32,
                        },
                        fan_out.target,
                    );
                }
            }
            AuxVertexKind::LoopTail => {
                let scc = tail_loop[&aux_index];
                let record = overlay.loops[scc.0 as usize]
                    .as_ref()
                    .expect("tail of a loop without a record");
                expansion.branch[node] = Some(AuxVar::LoopRepeat(scc));
                // Exit alternative (selector value 0). Its arc is the
                // collapsed loop's successor arc, which a chained construct
                // may have rewritten.
                if let Some(exit_fan_out) = vertex.fan_out.first() {
                    expansion.values[node].push(0);
                    add_arc(
                        &mut expansion,
                        node,
                        ArcId {
                            source: Vertex::Loop(scc),
                            index: 0,
                        },
                        exit_fan_out.target,
                    );
                }
                // Repetition alternative (selector value 1): the real
                // repetition arc back to the loop's entry point. An endless
                // loop's record has an empty fan-out, but the repetition
                // arc exists in the denoted CFG all the same.
                let repeat_target = match record.entry_demux {
                    Some(demux) => Vertex::Aux(demux),
                    None => Vertex::Block(record.entries[0]),
                };
                expansion.values[node].push(1);
                let repeat_dest = resolve(repeat_target);
                expansion.succs[node].push(repeat_dest);
            }
            AuxVertexKind::PromotedAssign { assignments } => {
                expansion.assigns[node] = assignments.to_vec();
                let fan_out = vertex.fan_out[0];
                add_arc(
                    &mut expansion,
                    node,
                    ArcId {
                        source: Vertex::Aux(
                            crate::llvm_parser::control_flow::overlay::AuxVertexId(
                                aux_index as u32,
                            ),
                        ),
                        index: 0,
                    },
                    fan_out.target,
                );
            }
        }
    }
    expansion
}

/// The paper's SHORTCIRCUITCFG, operationally: a node carrying constant
/// selector assignments whose successor branches on one of them retargets
/// directly to the taken alternative (dropping that assignment); nodes left
/// with no assignments and no branch are bypassed; what restructuring
/// inserted becomes unreachable and is ignored by the comparison.
fn short_circuit(expansion: &mut Expansion) {
    // Drain assignments through branches to a fixpoint. Each step removes
    // one assignment, so this terminates. The paper's null-vertex removal
    // runs in the SAME fixpoint as the substitution, so the branch is
    // looked up through any already-drained intermediaries (an assignment
    // cluster that emptied earlier is the paper's null vertex).
    loop {
        let mut changed = false;
        for node in 0..expansion.succs.len() {
            if expansion.assigns[node].is_empty() || expansion.succs[node].len() != 1 {
                continue;
            }
            let successor = final_target(expansion, expansion.succs[node][0]);
            let Some(branch_var) = expansion.branch[successor] else {
                continue;
            };
            let Some(position) = expansion.assigns[node]
                .iter()
                .rposition(|assign| assign.var == branch_var)
            else {
                continue;
            };
            let value = expansion.assigns[node][position].value;
            let Some(slot) = expansion.values[successor].iter().position(|&v| v == value) else {
                // No alternative for this selector value: leave it; the
                // comparison will report the faithfulness failure.
                continue;
            };
            expansion.succs[node][0] = expansion.succs[successor][slot];
            expansion.assigns[node].remove(position);
            changed = true;
        }
        if !changed {
            break;
        }
    }
}

/// Follow drained (empty, branch-free, single-successor) synthetic nodes to
/// the real destination.
fn final_target(expansion: &Expansion, mut node: Node) -> Node {
    let mut hops = 0;
    while node >= expansion.block_count
        && expansion.branch[node].is_none()
        && expansion.assigns[node].is_empty()
        && expansion.succs[node].len() == 1
    {
        node = expansion.succs[node][0];
        hops += 1;
        assert!(hops < 1_000_000, "bypass chain does not terminate");
    }
    node
}

/// Assert the paper's Corollary 5.7 for one function: short-circuiting the
/// expanded overlay recovers the original closed CFG exactly -- every
/// original block ends with the same ordered successor blocks it started
/// with, and nothing restructuring inserted remains on any path.
pub(super) fn assert_roundtrip(
    mapper: &BasicBlockMapper,
    tree: &SccTree,
    overlay: &Overlay,
    context: &str,
) {
    let mut expansion = expand(mapper, tree, overlay);
    short_circuit(&mut expansion);

    for index in 0..expansion.block_count {
        let block = BasicBlockId(index as u32);
        let expected: SmallVec<[usize; 4]> = if overlay.is_diverging(block) {
            // The closed-CFG convention: a diverging block's one arc leads
            // to the synthetic exit.
            SmallVec::from_slice(&[overlay.exit_block.0 as usize])
        } else {
            mapper
                .outputs(block)
                .iter()
                .map(|target| target.0 as usize)
                .collect()
        };
        let recovered: SmallVec<[usize; 4]> = expansion.succs[index]
            .iter()
            .map(|&succ| final_target(&expansion, succ))
            .collect();
        assert_eq!(
            expected[..],
            recovered[..],
            "{context}, block {index}: short-circuiting the restructured CFG did not \
             recover the original arcs (Corollary 5.7 violated). Recovered nodes at or \
             beyond {} are restructuring leftovers that failed to dissolve; leftover \
             details: {:?}",
            expansion.block_count,
            recovered
                .iter()
                .filter(|&&n| n >= expansion.block_count)
                .map(|&n| (n, expansion.branch[n], expansion.assigns[n].clone()))
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::control_flow::build_overlay;
    use crate::llvm_parser::scc::SccTree;
    use crate::rvsdg::{RVSDGMod, verify::RVSDGVerificationError};
    use llvm_ir::Module;
    use std::sync::Mutex;

    // llvm-ir lazily initialises a global attribute table on first parse
    // that races under concurrent test threads; serialise parses.
    static LLVM_PARSE_LOCK: Mutex<()> = Mutex::new(());

    /// Restructure every function of `module` and assert Corollary 5.7 for
    /// each: short-circuiting the expanded overlay recovers the original
    /// closed CFG.
    fn assert_module_roundtrip(module: &Module, context: &str) {
        for func in &module.functions {
            let mapper = crate::llvm_parser::intern_blocks_and_arcs(func);
            let diverging: Vec<bool> = func
                .basic_blocks
                .iter()
                .map(|b| matches!(b.term, llvm_ir::Terminator::Unreachable(_)))
                .chain(std::iter::once(false))
                .collect();
            let tree = SccTree::build(&mapper);
            let overlay = build_overlay(&mapper, &tree, diverging);
            assert_roundtrip(
                &mapper,
                &tree,
                &overlay,
                &format!("{context} fn {}", func.name),
            );
        }
    }

    fn roundtrip_ir(ir: &str) {
        let module = {
            let _guard = LLVM_PARSE_LOCK.lock().unwrap();
            Module::from_ir_str(ir).expect("parse test IR")
        };
        assert_module_roundtrip(&module, "inline IR");
    }

    #[test]
    fn roundtrip_structured_do_while() {
        roundtrip_ir(
            r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %header ]
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, 5
  br i1 %c, label %header, label %done
done:
  ret i32 %i.next
}
"#,
        );
    }

    #[test]
    fn roundtrip_multi_exit_loop_with_demux() {
        roundtrip_ir(
            r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %big = icmp sgt i32 %i, 3
  br i1 %big, label %exitA, label %latch
latch:
  %i.next = add i32 %i, 1
  %done = icmp sge i32 %i.next, 10
  br i1 %done, label %exitB, label %header
exitA:
  ret i32 100
exitB:
  ret i32 200
}
"#,
        );
    }

    #[test]
    fn roundtrip_irreducible_entry_demux() {
        roundtrip_ir(
            r#"
define i32 @f(i32 %0, i32 %1) {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %5
4:
  br label %8
5:
  br label %6
6:
  %.08 = phi i32 [ %16, %18 ], [ %1, %5 ]
  %.0 = phi i32 [ %.2, %18 ], [ 0, %5 ]
  %7 = add nsw i32 %.0, 10
  br label %8
8:
  %.19 = phi i32 [ %1, %4 ], [ %.08, %6 ]
  %.1 = phi i32 [ 0, %4 ], [ %7, %6 ]
  %9 = and i32 %.1, 1
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %13
11:
  %12 = add nsw i32 %.1, 100
  br label %15
13:
  %14 = add nsw i32 %.1, 1
  br label %15
15:
  %.2 = phi i32 [ %12, %11 ], [ %14, %13 ]
  %16 = sub nsw i32 %.19, 1
  %17 = icmp sgt i32 %16, 0
  br i1 %17, label %18, label %19
18:
  br label %6
19:
  ret i32 %.2
}
"#,
        );
    }

    #[test]
    fn roundtrip_break_out_of_nested_loops() {
        // The chained-exit shape (an inner exit arc that an outer pass
        // already rewrote): the fused cluster must drain through BOTH
        // loops' tails and demuxes back to the original arc.
        roundtrip_ir(
            r#"
define i32 @f(i32 %n) {
entry:
  br label %ho
ho:
  %i = phi i32 [ 0, %entry ], [ %i.next, %lo ]
  %co = icmp slt i32 %i, %n
  br i1 %co, label %hi, label %done
hi:
  %j = phi i32 [ 0, %ho ], [ %j.next, %li ]
  %s = add i32 %i, %j
  %brk = icmp eq i32 %s, 4
  br i1 %brk, label %done, label %li
li:
  %j.next = add i32 %j, 1
  %ci = icmp slt i32 %j.next, %n
  br i1 %ci, label %hi, label %lo
lo:
  %i.next = add i32 %i, 1
  br label %ho
done:
  %r = phi i32 [ %i, %ho ], [ %s, %hi ]
  ret i32 %r
}
"#,
        );
    }

    #[test]
    fn roundtrip_structured_loop_exit_into_demux() {
        // A structured do-while inside a switch arm whose successor arc is
        // an arm-boundary arc funneled through a continuation demux: the
        // loop's exit must fuse with the arm's continuation selector and
        // short-circuit back to the original single arc.
        roundtrip_ir(
            r#"
define i32 @f(i32 %x) {
entry:
  switch i32 %x, label %a [ i32 1, label %b
                            i32 2, label %c ]
a:
  br label %dw
dw:
  %j = phi i32 [ 0, %a ], [ %j2, %dw ]
  %j2 = add i32 %j, 1
  %q = icmp slt i32 %j2, 3
  br i1 %q, label %dw, label %join1
b:
  br label %join1
c:
  br label %join2
join1:
  br label %join2
join2:
  %r = phi i32 [ 1, %join1 ], [ 2, %c ]
  ret i32 %r
}
"#,
        );
    }

    #[test]
    fn roundtrip_trimming_promoted_cluster() {
        // The trimming shape: an arm's loop-exit cluster is promoted behind
        // the routing demux; short-circuiting must still recover the
        // original single arc.
        roundtrip_ir(
            r#"
define i32 @f(i32 %n, i32 %a, i32 %b) {
entry:
  br label %h
h:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %ch = icmp slt i32 %i, %n
  br i1 %ch, label %body, label %out
body:
  %c1 = icmp eq i32 %a, 0
  br i1 %c1, label %x, label %y
x:
  %c2 = icmp eq i32 %b, 0
  br i1 %c2, label %out2, label %j
y:
  br label %j
j:
  %v = phi i32 [ 1, %x ], [ 2, %y ]
  br label %latch
latch:
  %i.next = add i32 %i, 1
  br label %h
out:
  ret i32 0
out2:
  ret i32 %i
}
"#,
        );
    }

    /// The full-suite oracle: every C fixture, every function, both the
    /// Corollary 5.7 round-trip and predicate continuation form on the
    /// emitted RVSDG.
    #[test]
    fn all_c_fixtures_roundtrip_and_hold_predicate_form() {
        let fixtures = std::fs::read_dir("tests/fixtures/c").expect("fixture dir");
        let mut checked = 0usize;
        for entry in fixtures {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("c") {
                continue;
            }
            let module = {
                let _guard = LLVM_PARSE_LOCK.lock().unwrap();
                crate::c_file_to_mod(&path, &[], &[], true)
                    .unwrap_or_else(|e| panic!("compile {}: {e}", path.display()))
            };
            assert_module_roundtrip(&module, &path.display().to_string());

            let rvsdg = RVSDGMod::from_llvm_mod(module)
                .unwrap_or_else(|e| panic!("lower {}: {e}", path.display()));
            let predicate_errors: Vec<_> = rvsdg
                .verify()
                .into_iter()
                .filter(|err| {
                    matches!(
                        err,
                        RVSDGVerificationError::PredicateNonConditionUse(_)
                            | RVSDGVerificationError::PredicateUsedMoreThanOnce(..)
                    )
                })
                .collect();
            assert!(
                predicate_errors.is_empty(),
                "{}: predicate continuation form violated: {predicate_errors:?}",
                path.display()
            );
            checked += 1;
        }
        assert!(checked >= 40, "expected the fixture suite, found {checked}");
    }
}
