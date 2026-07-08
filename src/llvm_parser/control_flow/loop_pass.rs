//! **The loop restructuring pass** (Bahmann, Reissmann, Jahre, Meyer 2015,
//! section 4.1): turn every strongly connected component into a
//! single-entry, tail-controlled loop by writing overlay records, so that
//! the enclosing graph can treat the whole loop as one collapsed vertex and
//! the loop body is acyclic once its repetition arcs are set aside.
//!
//! For each component the pass classifies, against the CURRENT
//! overlay-applied graph (never the static per-component arc sets, which
//! know nothing about previously inserted machinery):
//!
//! - entry arcs: arcs entering the component from outside; their targets
//!   are the entry vertices;
//! - repetition arcs: arcs from inside the component to any entry vertex;
//! - exit arcs: arcs from inside the component to any vertex outside it.
//!
//! A component that is already a tail-controlled loop -- one entry vertex,
//! and one vertex carrying both the single repetition arc and the single
//! exit arc and nothing else -- is left alone apart from collapsing (its
//! entry arcs redirect to the collapsed vertex). Anything else is
//! restructured: every repetition and exit arc is funneled through an
//! inserted loop tail that branches on a repeat flag, an entry demux
//! routes multi-entry loops by an entry selector, and an exit demux routes
//! multi-exit loops by the same selector. The selector and flag are
//! assigned as constants on the rewritten arcs; an assignment is only
//! written when the demux that consumes it exists.
//!
//! Processing order is OUTERMOST-FIRST (an outer component is restructured
//! before the pass descends into its body), because restructuring the
//! outer loop creates arcs that inner classification must see. Sibling
//! components at one level run in topological order of the component
//! condensation, so a loop feeding another loop's entries is processed
//! first and the second loop classifies the first one's exit machinery
//! (its demux fan-out or collapsed successor arc) rather than original
//! arcs that have been rewritten away.

use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    control_flow::{
        overlay::{ArcId, AuxAssign, AuxVar, AuxVertexKind, LoopOverlay, Overlay, Vertex},
        view::{Membership, RegionView},
    },
    scc::{SccTree, SccTreeNodeId},
};

/// Run the loop pass over every component of `tree`, writing records into
/// `overlay`. After this, every loop presents as a collapsed vertex with
/// (at most) a single successor arc, and every loop body is acyclic under
/// the body view's repetition-arc hiding.
#[tracing::instrument(name = "loop_pass", skip_all, fields(blocks = mapper.blocks.len(), components = tree.len()))]
pub(in crate::llvm_parser) fn run_loop_pass(
    mapper: &BasicBlockMapper,
    tree: &SccTree,
    overlay: &mut Overlay,
) {
    let block_count = mapper.blocks.len();
    let mut pass = LoopPass {
        mapper,
        tree,
        overlay,
        in_component: vec![0; block_count],
        generation: 0,
    };
    pass.process_level(&tree.roots, None);
}

struct LoopPass<'a> {
    mapper: &'a BasicBlockMapper,
    tree: &'a SccTree,
    overlay: &'a mut Overlay,
    /// Scratch stamp set marking the blocks of the component currently
    /// being classified (block index to generation).
    in_component: Vec<u32>,
    generation: u32,
}

/// One entry arc: the arc and the entry vertex it targets.
struct EntryArc {
    arc: ArcId,
    entry: BasicBlockId,
}

/// One repetition arc: the arc and the entry vertex it repeats to.
struct RepetitionArc {
    arc: ArcId,
    entry: BasicBlockId,
}

/// One exit arc: the arc and its effective target outside the component
/// (a block, a sibling's demux, an enclosing loop's tail, or a collapsed
/// sibling loop).
struct ExitArc {
    arc: ArcId,
    target: Vertex,
}

/// Everything classification learns about one component before any record
/// is written.
struct Classification {
    entries: SmallVec<[BasicBlockId; 2]>,
    entry_arcs: SmallVec<[EntryArc; 4]>,
    repetition_arcs: SmallVec<[RepetitionArc; 4]>,
    exit_arcs: SmallVec<[ExitArc; 4]>,
    /// Distinct effective exit targets in ascending vertex order; the index
    /// is the selector value assigned when leaving toward that target.
    exit_targets: SmallVec<[Vertex; 2]>,
    /// True when the component is already a tail-controlled loop: one entry
    /// vertex, and one vertex whose only two arcs are the single repetition
    /// arc and the single exit arc.
    structured: bool,
}

impl LoopPass<'_> {
    /// Process the components of one nesting level, then descend into each
    /// component's body. `body_of` is the enclosing component when this
    /// level is a loop body (its repetition arcs are hidden from the
    /// classification views).
    fn process_level(&mut self, sccs: &[SccTreeNodeId], body_of: Option<SccTreeNodeId>) {
        if sccs.is_empty() {
            return;
        }
        // The level's collapse table: every sibling's blocks map to that
        // sibling. Only processed siblings present collapsed (the view
        // checks the overlay record exists), so the one table serves the
        // whole level while it is processed in order.
        let collapse = self.tree.collapse_table(sccs, self.in_component.len());

        for scc in self.sibling_topological_order(sccs) {
            self.restructure_component(scc, &collapse, body_of);
            let children = self.tree.children[scc.0 as usize].clone();
            self.process_level(&children, Some(scc));
        }
    }

    /// Sibling components ordered so that a component whose arcs feed
    /// another comes first. Distinct components cannot feed each other both
    /// ways (they would be one component), so the order exists; ties are
    /// broken by ascending component id for determinism.
    fn sibling_topological_order(&self, sccs: &[SccTreeNodeId]) -> SmallVec<[SccTreeNodeId; 4]> {
        let count = sccs.len();
        if count == 1 {
            return SmallVec::from_slice(sccs);
        }
        // Map each sibling's blocks to its local index.
        let mut sibling_of: Vec<Option<usize>> = vec![None; self.in_component.len()];
        for (local, &scc) in sccs.iter().enumerate() {
            for &block in &self.tree.blocks[scc.0 as usize] {
                sibling_of[block.0 as usize] = Some(local);
            }
        }
        // Edges over ORIGINAL block arcs suffice: every cross-sibling
        // feeding relationship is mediated by an original arc from one
        // component's block toward the other component (later machinery
        // only ever re-routes those same paths).
        let mut edges: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); count];
        let mut indegree: Vec<u32> = vec![0; count];
        for (local, &scc) in sccs.iter().enumerate() {
            for &block in &self.tree.blocks[scc.0 as usize] {
                for &succ in self.mapper.outputs(block) {
                    if let Some(other) = sibling_of[succ.0 as usize]
                        && other != local
                        && !edges[local].contains(&other)
                    {
                        edges[local].push(other);
                        indegree[other] += 1;
                    }
                }
            }
        }
        // Kahn's algorithm, taking ready components in ascending id order.
        let mut order: SmallVec<[SccTreeNodeId; 4]> = SmallVec::new();
        let mut ready: SmallVec<[usize; 4]> =
            (0..count).filter(|&local| indegree[local] == 0).collect();
        while !ready.is_empty() {
            ready.sort_unstable_by_key(|&local| sccs[local].0);
            let local = ready.remove(0);
            order.push(sccs[local]);
            for &next in &edges[local] {
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    ready.push(next);
                }
            }
        }
        debug_assert_eq!(order.len(), count, "sibling components formed a cycle");
        order
    }

    /// Classify one component against the current overlay-applied graph and
    /// write its records.
    fn restructure_component(
        &mut self,
        scc: SccTreeNodeId,
        collapse: &[Option<SccTreeNodeId>],
        body_of: Option<SccTreeNodeId>,
    ) {
        self.generation += 1;
        let generation = self.generation;
        let mut blocks: SmallVec<[BasicBlockId; 8]> = self.tree.blocks[scc.0 as usize].clone();
        blocks.sort_unstable();
        for &block in &blocks {
            self.in_component[block.0 as usize] = generation;
        }

        let classification = classify(
            self.mapper,
            self.overlay,
            collapse,
            body_of,
            &blocks,
            &self.in_component,
            generation,
        );

        if classification.structured {
            let back_edge = classification.repetition_arcs[0].arc;
            self.overlay.loops[scc.0 as usize] = Some(LoopOverlay {
                entries: classification.entries,
                exit_targets: classification.exit_targets,
                entry_demux: None,
                tail: None,
                exit_demux: None,
                structured_back_edge: Some(back_edge),
            });
            // Collapsing still needs the entry arcs to enter the collapsed
            // vertex; a structured loop's other arcs stay untouched (its
            // exit arc is presented as the collapsed vertex's successor by
            // the view).
            for entry_arc in &classification.entry_arcs {
                self.overlay
                    .rewrite_arc(entry_arc.arc, &[], Vertex::Loop(scc));
            }
            return;
        }

        let entries = classification.entries;
        let exit_targets = classification.exit_targets;
        let entry_demux = (entries.len() > 1).then(|| {
            let targets: SmallVec<[Vertex; 4]> =
                entries.iter().map(|&entry| Vertex::Block(entry)).collect();
            self.overlay
                .add_aux_vertex(AuxVertexKind::LoopEntryDemux { scc }, &targets, Some(scc))
        });
        let exit_demux = (exit_targets.len() > 1).then(|| {
            self.overlay.add_aux_vertex(
                AuxVertexKind::LoopExitDemux { scc },
                &exit_targets,
                body_of,
            )
        });
        // The loop tail: alternative 0 leaves, alternative 1 repeats. An
        // endless loop (no exit arcs) gets an empty fan-out: its repeat is
        // implicit in the theta and nothing ever leaves through it.
        let tail_fan_out: SmallVec<[Vertex; 2]> = if exit_targets.is_empty() {
            SmallVec::new()
        } else {
            let exit_destination = match exit_demux {
                Some(demux) => Vertex::Aux(demux),
                None => exit_targets[0],
            };
            let repeat_destination = match entry_demux {
                Some(demux) => Vertex::Aux(demux),
                None => Vertex::Block(entries[0]),
            };
            SmallVec::from_slice(&[exit_destination, repeat_destination])
        };
        let tail = self
            .overlay
            .add_aux_vertex(AuxVertexKind::LoopTail, &tail_fan_out, Some(scc));

        // The record goes in before the rewrites so the component presents
        // collapsed to every later query at this level.
        self.overlay.loops[scc.0 as usize] = Some(LoopOverlay {
            entries: entries.clone(),
            exit_targets: exit_targets.clone(),
            entry_demux,
            tail: Some(tail),
            exit_demux,
            structured_back_edge: None,
        });

        let selector = AuxVar::LoopVertexSelector(scc);
        let repeat = AuxVar::LoopRepeat(scc);
        let entry_index = |entry: BasicBlockId| {
            entries
                .iter()
                .position(|&e| e == entry)
                .expect("repetition/entry arc targets a classified entry vertex") as u32
        };

        for entry_arc in &classification.entry_arcs {
            let mut assignments: SmallVec<[AuxAssign; 2]> = SmallVec::new();
            if entry_demux.is_some() {
                assignments.push(AuxAssign {
                    var: selector,
                    value: entry_index(entry_arc.entry),
                });
            }
            self.overlay
                .rewrite_arc(entry_arc.arc, &assignments, Vertex::Loop(scc));
        }
        for repetition_arc in &classification.repetition_arcs {
            let mut assignments: SmallVec<[AuxAssign; 2]> = SmallVec::new();
            if entry_demux.is_some() {
                assignments.push(AuxAssign {
                    var: selector,
                    value: entry_index(repetition_arc.entry),
                });
            }
            assignments.push(AuxAssign {
                var: repeat,
                value: 1,
            });
            self.overlay
                .rewrite_arc(repetition_arc.arc, &assignments, Vertex::Aux(tail));
        }
        for exit_arc in &classification.exit_arcs {
            let mut assignments: SmallVec<[AuxAssign; 2]> = SmallVec::new();
            if exit_demux.is_some() {
                let value = exit_targets
                    .iter()
                    .position(|&t| t == exit_arc.target)
                    .expect("exit arc targets a classified exit target")
                    as u32;
                assignments.push(AuxAssign {
                    var: selector,
                    value,
                });
            }
            assignments.push(AuxAssign {
                var: repeat,
                value: 0,
            });
            self.overlay
                .rewrite_arc(exit_arc.arc, &assignments, Vertex::Aux(tail));
        }

        // A restructured loop's successor arc (to its exit demux, or to its
        // sole exit target) is implicit in traversal; register it so later
        // in-arc queries -- sibling loops classifying their entry arcs, and
        // partitioning counting in-region indegrees -- see the collapsed
        // loop enter its successor. (A structured loop's unrewritten exit
        // arc is presented by the view instead; an endless loop has no
        // successor.)
        match exit_demux {
            Some(demux) => self
                .overlay
                .register_loop_successor(scc, Vertex::Aux(demux)),
            None => {
                if let Some(&target) = exit_targets.first() {
                    self.overlay.register_loop_successor(scc, target);
                }
            }
        }
    }
}

/// Classify `blocks` (one component, stamped in `in_component` with
/// `generation`) against the current overlay-applied graph.
fn classify(
    mapper: &BasicBlockMapper,
    overlay: &Overlay,
    collapse: &[Option<SccTreeNodeId>],
    body_of: Option<SccTreeNodeId>,
    blocks: &[BasicBlockId],
    in_component: &[u32],
    generation: u32,
) -> Classification {
    let inside = |vertex: Vertex| match vertex {
        Vertex::Block(b) => in_component[b.0 as usize] == generation,
        // Aux and collapsed-loop vertices always belong to already-processed
        // machinery outside this component.
        Vertex::Aux(_) | Vertex::Loop(_) => false,
    };
    let view = RegionView {
        mapper,
        overlay,
        members: Membership::Universal,
        collapse,
        body_of,
    };

    let mut entries: SmallVec<[BasicBlockId; 2]> = SmallVec::new();
    let mut entry_arcs: SmallVec<[EntryArc; 4]> = SmallVec::new();
    for &block in blocks {
        let mut is_entry = false;
        for incoming in view.arcs_in(Vertex::Block(block)) {
            if !inside(incoming.arc.source) {
                is_entry = true;
                entry_arcs.push(EntryArc {
                    arc: incoming.arc,
                    entry: block,
                });
            }
        }
        if is_entry {
            entries.push(block);
        }
    }
    // `blocks` is sorted, so `entries` is in ascending block order: the
    // selector numbering convention.
    debug_assert!(entries.windows(2).all(|w| w[0] < w[1]));

    let mut repetition_arcs: SmallVec<[RepetitionArc; 4]> = SmallVec::new();
    let mut exit_arcs: SmallVec<[ExitArc; 4]> = SmallVec::new();
    for &block in blocks {
        for outgoing in view.arcs_out(Vertex::Block(block)) {
            match outgoing.target {
                Vertex::Block(target) if in_component[target.0 as usize] == generation => {
                    if entries.contains(&target) {
                        repetition_arcs.push(RepetitionArc {
                            arc: outgoing.arc,
                            entry: target,
                        });
                    }
                }
                target => {
                    exit_arcs.push(ExitArc {
                        arc: outgoing.arc,
                        target,
                    });
                }
            }
        }
    }

    let mut exit_targets: SmallVec<[Vertex; 2]> = SmallVec::new();
    for exit_arc in &exit_arcs {
        if !exit_targets.contains(&exit_arc.target) {
            exit_targets.push(exit_arc.target);
        }
    }
    exit_targets.sort_unstable();

    // Already a tail-controlled loop? One entry vertex, and one vertex
    // whose only two arcs are the single repetition arc and the single
    // exit arc (so its branch condition can serve as the theta repetition
    // predicate directly).
    let structured = entries.len() == 1
        && repetition_arcs.len() == 1
        && exit_arcs.len() == 1
        && repetition_arcs[0].arc.source == exit_arcs[0].arc.source
        && view.arcs_out(repetition_arcs[0].arc.source).len() == 2;

    Classification {
        entries,
        entry_arcs,
        repetition_arcs,
        exit_arcs,
        exit_targets,
        structured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::scc::SccTree;
    use llvm_ir::Module;
    use std::sync::Mutex;

    // llvm-ir lazily initialises a global attribute table on first parse
    // that races under concurrent test threads; serialise parses.
    static LLVM_PARSE_LOCK: Mutex<()> = Mutex::new(());

    struct Prepared {
        mapper: BasicBlockMapper,
        tree: SccTree,
        overlay: Overlay,
    }

    /// Parse `ir`, intern the first function's blocks, build the component
    /// tree, and run the loop pass.
    fn run_on(ir: &str) -> Prepared {
        let module = {
            let _guard = LLVM_PARSE_LOCK.lock().unwrap();
            Module::from_ir_str(ir).expect("parse test IR")
        };
        let func = &module.functions[0];
        let mapper = crate::llvm_parser::intern_blocks_and_arcs(func);
        let diverging = func
            .basic_blocks
            .iter()
            .map(|b| matches!(b.term, llvm_ir::Terminator::Unreachable(_)))
            // The synthetic exit block never diverges.
            .chain(std::iter::once(false))
            .collect();
        let tree = SccTree::build(&mapper);
        let mut overlay = Overlay::new(&mapper, diverging, tree.len());
        run_loop_pass(&mapper, &tree, &mut overlay);
        Prepared {
            mapper,
            tree,
            overlay,
        }
    }

    fn block_id(prepared: &Prepared, name: &str) -> BasicBlockId {
        *prepared
            .mapper
            .get(&llvm_ir::Name::Name(Box::new(name.to_string())))
            .unwrap_or_else(|| panic!("no block named {name}"))
    }

    fn sole_loop(prepared: &Prepared) -> (&LoopOverlay, SccTreeNodeId) {
        assert_eq!(1, prepared.tree.len(), "expected exactly one loop");
        (
            prepared.overlay.loops[0].as_ref().expect("loop record"),
            SccTreeNodeId(0),
        )
    }

    #[test]
    fn do_while_is_structured_and_only_collapsed() {
        let ir = r#"
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
"#;
        let prepared = run_on(ir);
        let (record, scc) = sole_loop(&prepared);
        let header = block_id(&prepared, "header");
        let done = block_id(&prepared, "done");

        assert_eq!(&[header][..], &record.entries[..]);
        assert_eq!(&[Vertex::Block(done)][..], &record.exit_targets[..]);
        assert!(record.tail.is_none(), "do-while is already structured");
        assert!(record.entry_demux.is_none());
        assert!(record.exit_demux.is_none());
        // The back edge is the header's true alternative (arc index 0).
        assert_eq!(
            Some(ArcId {
                source: Vertex::Block(header),
                index: 0,
            }),
            record.structured_back_edge
        );
        // The entry arc collapses into the loop vertex with no assignments.
        let entry = block_id(&prepared, "entry");
        let rewrite = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(entry),
                index: 0,
            })
            .expect("entry arc rewritten");
        assert!(rewrite.assignments.is_empty());
        assert_eq!(Vertex::Loop(scc), rewrite.redirect);
    }

    #[test]
    fn test_first_while_gets_a_tail_and_repeat_flags() {
        let ir = r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %c = icmp slt i32 %i, 5
  br i1 %c, label %body, label %done
body:
  %i.next = add i32 %i, 1
  br label %header
done:
  ret i32 %i
}
"#;
        let prepared = run_on(ir);
        let (record, scc) = sole_loop(&prepared);
        let header = block_id(&prepared, "header");
        let body = block_id(&prepared, "body");
        let done = block_id(&prepared, "done");

        assert_eq!(&[header][..], &record.entries[..]);
        assert_eq!(&[Vertex::Block(done)][..], &record.exit_targets[..]);
        let tail = record.tail.expect("test-first loop needs a tail");
        assert!(record.entry_demux.is_none(), "single entry");
        assert!(record.exit_demux.is_none(), "single exit target");

        // The repetition arc (body -> header) carries repeat := 1 and
        // funnels through the tail; no selector, since no demux consumes it.
        let repetition = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(body),
                index: 0,
            })
            .expect("repetition arc rewritten");
        assert_eq!(
            &[AuxAssign {
                var: AuxVar::LoopRepeat(scc),
                value: 1,
            }][..],
            &repetition.assignments[..]
        );
        assert_eq!(Vertex::Aux(tail), repetition.redirect);

        // The exit arc (header -> done, false alternative = index 1)
        // carries repeat := 0.
        let exit = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(header),
                index: 1,
            })
            .expect("exit arc rewritten");
        assert_eq!(
            &[AuxAssign {
                var: AuxVar::LoopRepeat(scc),
                value: 0,
            }][..],
            &exit.assignments[..]
        );
        assert_eq!(Vertex::Aux(tail), exit.redirect);

        // The collapsed loop's successor arc is registered at the exit
        // target for later passes' in-arc classification.
        assert!(
            prepared
                .overlay
                .overlay_in_arcs(Vertex::Block(done))
                .contains(&ArcId {
                    source: Vertex::Loop(scc),
                    index: 0,
                })
        );
    }

    #[test]
    fn multi_exit_loop_gets_an_exit_demux_with_selectors() {
        let ir = r#"
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
"#;
        let prepared = run_on(ir);
        let (record, scc) = sole_loop(&prepared);
        let header = block_id(&prepared, "header");
        let latch = block_id(&prepared, "latch");
        let exit_a = block_id(&prepared, "exitA");
        let exit_b = block_id(&prepared, "exitB");

        let exit_demux = record.exit_demux.expect("two exit targets");
        let expected_targets: SmallVec<[Vertex; 2]> = {
            let mut t: SmallVec<[Vertex; 2]> =
                SmallVec::from_slice(&[Vertex::Block(exit_a), Vertex::Block(exit_b)]);
            t.sort_unstable();
            t
        };
        assert_eq!(expected_targets, record.exit_targets);

        // Each exit arc carries selector := its target's index and
        // repeat := 0. header -> exitA is the true alternative (index 0).
        let selector = AuxVar::LoopVertexSelector(scc);
        let exit_a_index = record
            .exit_targets
            .iter()
            .position(|&t| t == Vertex::Block(exit_a))
            .unwrap() as u32;
        let via_header = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(header),
                index: 0,
            })
            .expect("header exit arc rewritten");
        assert_eq!(
            &[
                AuxAssign {
                    var: selector,
                    value: exit_a_index,
                },
                AuxAssign {
                    var: AuxVar::LoopRepeat(scc),
                    value: 0,
                },
            ][..],
            &via_header.assignments[..]
        );

        let exit_b_index = record
            .exit_targets
            .iter()
            .position(|&t| t == Vertex::Block(exit_b))
            .unwrap() as u32;
        let via_latch = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(latch),
                index: 0,
            })
            .expect("latch exit arc rewritten");
        assert_eq!(exit_b_index, via_latch.assignments[0].value);

        // The demux fan-out arcs enter the exit targets.
        assert!(
            prepared
                .overlay
                .overlay_in_arcs(Vertex::Block(exit_a))
                .iter()
                .any(|a| a.source == Vertex::Aux(exit_demux))
        );
    }

    #[test]
    fn irreducible_loop_gets_an_entry_demux_and_entry_selectors() {
        // Two entries (blocks 6 and 8) reached from an initial branch; the
        // shape of tests/fixtures/c/08_irreducible.c.
        let ir = r#"
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
"#;
        let prepared = run_on(ir);
        let (record, scc) = sole_loop(&prepared);
        let six = *prepared.mapper.get(&llvm_ir::Name::Number(6)).unwrap();
        let eight = *prepared.mapper.get(&llvm_ir::Name::Number(8)).unwrap();

        assert_eq!(&[six, eight][..], &record.entries[..]);
        let entry_demux = record.entry_demux.expect("two entries");
        record.tail.expect("irreducible loop is restructured");

        // Entry arc 4 -> 8 assigns the selector for entry index 1 (block 8)
        // and collapses into the loop.
        let four = *prepared.mapper.get(&llvm_ir::Name::Number(4)).unwrap();
        let selector = AuxVar::LoopVertexSelector(scc);
        let entry_arc = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(four),
                index: 0,
            })
            .expect("entry arc rewritten");
        assert_eq!(
            &[AuxAssign {
                var: selector,
                value: 1,
            }][..],
            &entry_arc.assignments[..]
        );
        assert_eq!(Vertex::Loop(scc), entry_arc.redirect);

        // The repetition arc 18 -> 6 assigns selector := 0 (entry block 6)
        // and repeat := 1.
        let eighteen = *prepared.mapper.get(&llvm_ir::Name::Number(18)).unwrap();
        let repetition = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(eighteen),
                index: 0,
            })
            .expect("repetition arc rewritten");
        assert_eq!(
            &[
                AuxAssign {
                    var: selector,
                    value: 0,
                },
                AuxAssign {
                    var: AuxVar::LoopRepeat(scc),
                    value: 1,
                },
            ][..],
            &repetition.assignments[..]
        );

        // The entry demux fans out to both entries.
        let demux = &prepared.overlay.aux_vertices[entry_demux.0 as usize];
        assert_eq!(2, demux.fan_out.len());
        assert_eq!(Vertex::Block(six), demux.fan_out[0].target);
        assert_eq!(Vertex::Block(eight), demux.fan_out[1].target);
    }

    #[test]
    fn nested_structured_loops_collapse_inner_within_outer_body() {
        let ir = r#"
define i32 @f() {
entry:
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %j.next = add i32 %j, 1
  %jc = icmp slt i32 %j.next, 3
  br i1 %jc, label %inner, label %latch
latch:
  %i.next = add i32 %i, 1
  %ic = icmp slt i32 %i.next, 4
  br i1 %ic, label %outer, label %done
done:
  ret i32 %i.next
}
"#;
        let prepared = run_on(ir);
        assert_eq!(2, prepared.tree.len(), "outer loop plus nested inner");
        let outer_scc = prepared.tree.roots[0];
        let inner_scc = prepared.tree.children[outer_scc.0 as usize][0];
        let outer = prepared.overlay.loops[outer_scc.0 as usize]
            .as_ref()
            .expect("outer record");
        let inner = prepared.overlay.loops[inner_scc.0 as usize]
            .as_ref()
            .expect("inner record");

        // Both are tail-controlled already: no aux machinery, back edges
        // recorded, entry arcs collapsed.
        assert!(outer.tail.is_none());
        assert!(inner.tail.is_none());
        let inner_block = block_id(&prepared, "inner");
        assert_eq!(&[inner_block][..], &inner.entries[..]);

        // The outer body's arc into the inner loop (outer -> inner) is the
        // inner loop's entry arc, redirected into the collapsed inner
        // vertex.
        let outer_block = block_id(&prepared, "outer");
        let entry_arc = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(outer_block),
                index: 0,
            })
            .expect("inner entry arc rewritten");
        assert_eq!(Vertex::Loop(inner_scc), entry_arc.redirect);
    }

    #[test]
    fn chained_loops_rewrite_the_first_loops_successor_arc() {
        // Loop A (test-first, restructured) exits straight into loop B's
        // header. B's entry classification must find and rewrite A's
        // collapsed successor arc, not the original (already rewritten)
        // exit arc.
        let ir = r#"
define i32 @f() {
entry:
  br label %ha
ha:
  %i = phi i32 [ 0, %entry ], [ %i.next, %ba ]
  %ca = icmp slt i32 %i, 5
  br i1 %ca, label %ba, label %hb
ba:
  %i.next = add i32 %i, 1
  br label %ha
hb:
  %j = phi i32 [ %i, %ha ], [ %j.next, %hb ]
  %j.next = add i32 %j, 1
  %cb = icmp slt i32 %j.next, 9
  br i1 %cb, label %hb, label %done
done:
  ret i32 %j.next
}
"#;
        let prepared = run_on(ir);
        assert_eq!(2, prepared.tree.len());
        // Identify which component is which by entry block.
        let ha = block_id(&prepared, "ha");
        let hb = block_id(&prepared, "hb");
        let scc_of = |entry: BasicBlockId| {
            (0..prepared.tree.len() as u32)
                .map(SccTreeNodeId)
                .find(|&scc| {
                    prepared.overlay.loops[scc.0 as usize]
                        .as_ref()
                        .is_some_and(|record| record.entries.contains(&entry))
                })
                .expect("component for entry")
        };
        let scc_a = scc_of(ha);
        let scc_b = scc_of(hb);

        // A's collapsed successor arc was rewritten by B's entry pass to
        // enter B's collapsed vertex.
        let successor = ArcId {
            source: Vertex::Loop(scc_a),
            index: 0,
        };
        let rewrite = prepared
            .overlay
            .rewrite_of(successor)
            .expect("B rewrote A's successor arc");
        assert_eq!(Vertex::Loop(scc_b), rewrite.redirect);
        assert!(
            rewrite.assignments.is_empty(),
            "B has a single entry: no selector"
        );
        // And the in-arc bookkeeping moved it from hb to Loop(B).
        assert!(
            prepared
                .overlay
                .overlay_in_arcs(Vertex::Loop(scc_b))
                .contains(&successor)
        );
        assert!(
            !prepared
                .overlay
                .overlay_in_arcs(Vertex::Block(hb))
                .contains(&successor)
        );
    }

    #[test]
    fn endless_loop_gets_a_tail_with_no_exits() {
        let ir = r#"
define i32 @f() {
entry:
  br label %spin
spin:
  br label %spin
}
"#;
        let prepared = run_on(ir);
        let (record, scc) = sole_loop(&prepared);
        assert!(record.exit_targets.is_empty());
        let tail = record.tail.expect("endless loop is not tail-controlled");
        assert!(
            prepared.overlay.aux_vertices[tail.0 as usize]
                .fan_out
                .is_empty()
        );
        // The repetition arc still funnels through the tail with
        // repeat := 1, so the theta predicate exists and is constant.
        let spin = block_id(&prepared, "spin");
        let repetition = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(spin),
                index: 0,
            })
            .expect("repetition arc rewritten");
        assert_eq!(
            &[AuxAssign {
                var: AuxVar::LoopRepeat(scc),
                value: 1,
            }][..],
            &repetition.assignments[..]
        );
    }
}
