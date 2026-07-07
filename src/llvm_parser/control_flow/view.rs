//! **The region view**: traversal over the virtually restructured control
//! flow graph -- the immutable interned CFG composed with the
//! [`Overlay`](super::overlay::Overlay)'s records. Every restructuring pass
//! and the RVSDG emitter walk regions through this view; nothing ever walks
//! the raw block graph directly once restructuring begins.
//!
//! A region is a closed subgraph: the whole function, a loop body, one
//! alternative of a branch, or a tail. The view scopes traversal to the
//! region's member set and presents the graph AS RESTRUCTURED: rewritten
//! arcs carry their auxiliary assignments and enter their redirect target,
//! inserted vertices expose their fan-out, and a processed loop appears as
//! a single collapsed vertex.
//!
//! Traversing an arc has a fixed meaning for the emitter (order matters):
//! first apply the arc's phi copies, then its auxiliary assignments, then
//! control moves to the target. Phi copies are PARALLEL copies -- resolve
//! every incoming operand against the symbol table as it stood before the
//! arc, then write all destinations -- because a block's phis may reference
//! each other's destinations to mean the previous iteration's values.

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    control_flow::overlay::{ArcId, AuxAssign, Overlay, Vertex},
    scc::SccTreeNodeId,
};

/// A region's member set, in whichever representation the pass at hand
/// keeps: the restructuring passes use reusable stamp sets, the emitter
/// uses owned hash sets built from partition results, and the function
/// root is universal.
#[derive(Clone, Copy)]
pub(in crate::llvm_parser) enum Membership<'a> {
    Universal,
    Stamps(&'a VertexSet),
    Set(&'a FxHashSet<Vertex>),
}

impl Membership<'_> {
    pub(in crate::llvm_parser) fn contains(&self, vertex: Vertex) -> bool {
        match self {
            Membership::Universal => true,
            Membership::Stamps(set) => set.contains(vertex),
            Membership::Set(set) => set.contains(&vertex),
        }
    }
}

/// The phi copies a traversed arc performs: bind `block`'s phi destinations
/// to their incoming values for predecessor `from`. For a rewritten arc,
/// `block` is the arc's ORIGINAL target and `from` its ORIGINAL source: the
/// redirect moves where control goes next, never where the phi values come
/// from. When `block` is the synthetic exit block, the "phi" is the
/// distinguished return-value symbol, bound from `from`'s return operand
/// (or poison if `from` diverges). Arcs created by the restructuring itself
/// (aux fan-out arcs, collapsed-loop successor arcs) carry no phi copies:
/// their targets' phis were already bound when the incoming rewritten arcs
/// were traversed, and the bound values travel onward as ordinary symbols.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::llvm_parser) struct PhiCopies {
    pub from: BasicBlockId,
    pub block: BasicBlockId,
}

/// One traversed arc, overlay applied.
#[derive(Debug)]
pub(in crate::llvm_parser) struct TraversedArc<'a> {
    /// The arc's identity, for passes that rewrite the arcs they classify.
    pub arc: ArcId,
    pub phi_copies: Option<PhiCopies>,
    /// The auxiliary assignments the arc carries, in evaluation order.
    pub assignments: &'a [AuxAssign],
    /// The vertex the arc enters, as seen at this view's nesting level (a
    /// target inside a processed loop presents as the collapsed vertex).
    pub target: Vertex,
}

/// A reusable stamp set over the three vertex id spaces. Clearing is O(1)
/// (bump the generation); the backing arrays are allocated once per
/// function and grown only when restructuring inserts aux vertices.
#[derive(Debug)]
pub(in crate::llvm_parser) struct VertexSet {
    blocks: Vec<u32>,
    aux: Vec<u32>,
    loops: Vec<u32>,
    generation: u32,
}

impl VertexSet {
    pub(in crate::llvm_parser) fn new(block_count: usize, loop_count: usize) -> Self {
        Self {
            blocks: vec![0; block_count],
            aux: Vec::new(),
            loops: vec![0; loop_count],
            generation: 1,
        }
    }

    /// Empty the set in O(1) by moving to a fresh generation.
    pub(in crate::llvm_parser) fn clear(&mut self) {
        self.generation += 1;
    }

    pub(in crate::llvm_parser) fn insert(&mut self, vertex: Vertex) {
        let generation = self.generation;
        *self.slot_mut(vertex) = generation;
    }

    pub(in crate::llvm_parser) fn contains(&self, vertex: Vertex) -> bool {
        self.slot(vertex) == Some(self.generation)
    }

    fn slot(&self, vertex: Vertex) -> Option<u32> {
        match vertex {
            Vertex::Block(b) => self.blocks.get(b.0 as usize).copied(),
            Vertex::Aux(a) => self.aux.get(a.0 as usize).copied(),
            Vertex::Loop(l) => self.loops.get(l.0 as usize).copied(),
        }
    }

    fn slot_mut(&mut self, vertex: Vertex) -> &mut u32 {
        match vertex {
            Vertex::Block(b) => &mut self.blocks[b.0 as usize],
            Vertex::Aux(a) => {
                let index = a.0 as usize;
                if index >= self.aux.len() {
                    self.aux.resize(index + 1, 0);
                }
                &mut self.aux[index]
            }
            Vertex::Loop(l) => &mut self.loops[l.0 as usize],
        }
    }
}

/// A closed subgraph of the virtually restructured CFG, scoped for one
/// pass or one emitter recursion level.
pub(in crate::llvm_parser) struct RegionView<'a> {
    pub mapper: &'a BasicBlockMapper,
    pub overlay: &'a Overlay,
    /// Region membership. The function root is universal; sub-region member
    /// sets come from partitioning (alternative subgraphs) and the loop
    /// pass (body sets).
    pub members: Membership<'a>,
    /// Per block: the processed loop this block is collapsed into AT THIS
    /// VIEW'S NESTING LEVEL, if any. The function root's table holds the
    /// outermost components; a loop body's table holds that component's
    /// direct children. Only loops the loop pass has already processed
    /// (their overlay record exists) are presented collapsed.
    pub collapse: &'a [Option<SccTreeNodeId>],
    /// Set when this view is a loop body: that loop's repetition arcs are
    /// hidden from traversal, because the body must be acyclic and must
    /// terminate at its tail. Hiding removes control flow, not payload: a
    /// structured loop's back-edge phi copies still apply at the body's
    /// end (see [`Self::hidden_back_edge`]).
    pub body_of: Option<SccTreeNodeId>,
}

impl<'a> RegionView<'a> {
    pub(in crate::llvm_parser) fn contains(&self, vertex: Vertex) -> bool {
        self.members.contains(vertex)
    }

    /// Outgoing arcs of `vertex` in alternative order, rewrites applied and
    /// this loop body's repetition arcs hidden.
    pub(in crate::llvm_parser) fn arcs_out(
        &self,
        vertex: Vertex,
    ) -> SmallVec<[TraversedArc<'a>; 4]> {
        let mut arcs: SmallVec<[TraversedArc<'a>; 4]> = SmallVec::new();
        match vertex {
            Vertex::Block(block) => {
                let arc_count = if self.overlay.is_diverging(block) {
                    // The closed-CFG convention: a diverging block gets one
                    // synthetic arc to the exit block (arc index 0).
                    1
                } else {
                    self.mapper.outputs(block).len()
                };
                for index in 0..arc_count {
                    let arc = ArcId {
                        source: vertex,
                        index: index as u32,
                    };
                    if self.is_hidden_repetition_arc(arc) {
                        continue;
                    }
                    arcs.push(self.traverse(arc));
                }
            }
            Vertex::Aux(aux) => {
                let fan_out_len = self.overlay.aux_vertices[aux.0 as usize].fan_out.len();
                for index in 0..fan_out_len {
                    let arc = ArcId {
                        source: vertex,
                        index: index as u32,
                    };
                    if self.is_hidden_repetition_arc(arc) {
                        continue;
                    }
                    arcs.push(self.traverse(arc));
                }
            }
            Vertex::Loop(scc) => {
                let loop_overlay = self.overlay.loops[scc.0 as usize]
                    .as_ref()
                    .expect("collapsed loop traversed before its loop pass ran");
                // A collapsed loop has at most one successor arc: to its
                // exit demux, or to its sole exit target. Endless loops
                // have none.
                let has_successor =
                    loop_overlay.exit_demux.is_some() || loop_overlay.exit_targets.len() == 1;
                if has_successor {
                    arcs.push(self.traverse(ArcId {
                        source: vertex,
                        index: 0,
                    }));
                }
            }
        }
        arcs
    }

    /// Incoming arcs of `vertex` under the overlay: original arcs whose
    /// effective target is still `vertex` (an original arc from inside a
    /// collapsed structured loop presents as that loop's successor arc),
    /// plus the arcs the overlay routes here (rewritten redirects and aux
    /// fan-out arcs).
    pub(in crate::llvm_parser) fn arcs_in(
        &self,
        vertex: Vertex,
    ) -> SmallVec<[TraversedArc<'a>; 4]> {
        let mut arcs: SmallVec<[TraversedArc<'a>; 4]> = SmallVec::new();
        if let Vertex::Block(block) = vertex {
            for &(source, index) in self.overlay.original_in_arcs(block) {
                let arc = ArcId {
                    source: Vertex::Block(source),
                    index,
                };
                if self.overlay.rewrite_of(arc).is_some() {
                    // Redirected away; if it now enters `vertex` again it is
                    // in the overlay in-arc list below.
                    continue;
                }
                if self.is_hidden_repetition_arc(arc) {
                    continue;
                }
                // An unrewritten arc out of a processed loop is that
                // (structured) loop's single exit arc; at this level it is
                // the collapsed vertex's successor arc. If THAT arc has been
                // rewritten (a later construct redirected the loop's
                // successor), it no longer enters `vertex` and is listed in
                // its redirect target's overlay in-arcs instead.
                let arc = match self.collapsed_loop_of(source) {
                    Some(scc) => {
                        let successor = ArcId {
                            source: Vertex::Loop(scc),
                            index: 0,
                        };
                        if self.overlay.rewrite_of(successor).is_some() {
                            continue;
                        }
                        successor
                    }
                    None => arc,
                };
                arcs.push(self.traverse(arc));
            }
        }
        for &arc in self.overlay.overlay_in_arcs(vertex) {
            if self.is_hidden_repetition_arc(arc) {
                continue;
            }
            arcs.push(self.traverse(arc));
        }
        arcs
    }

    /// A structured loop body's hidden back edge, for the emitter: its phi
    /// copies define the loop variables' next-iteration values and must be
    /// applied at the body's end even though the arc itself is hidden from
    /// control flow. `None` for restructured loops (their repetition arcs
    /// are rewritten toward the tail and traversed normally mid-body) and
    /// outside body views.
    pub(in crate::llvm_parser) fn hidden_back_edge(&self) -> Option<TraversedArc<'a>> {
        let scc = self.body_of?;
        let loop_overlay = self.overlay.loops[scc.0 as usize].as_ref()?;
        let arc = loop_overlay.structured_back_edge?;
        let Vertex::Block(source) = arc.source else {
            unreachable!("a structured loop's back edge is an original block arc");
        };
        debug_assert!(
            self.overlay.rewrite_of(arc).is_none(),
            "a structured loop has no rewrites"
        );
        let target = self.mapper.outputs(source)[arc.index as usize];
        Some(TraversedArc {
            arc,
            phi_copies: Some(PhiCopies {
                from: source,
                block: target,
            }),
            assignments: &[],
            target: Vertex::Block(target),
        })
    }

    /// Resolve one arc: raw target, rewrite application, phi copies, and
    /// collapse presentation.
    fn traverse(&self, arc: ArcId) -> TraversedArc<'a> {
        let (raw_target, phi_copies) = match arc.source {
            Vertex::Block(source) => {
                let raw_target = if self.overlay.is_diverging(source) {
                    debug_assert_eq!(arc.index, 0);
                    self.overlay.exit_block
                } else {
                    self.mapper.outputs(source)[arc.index as usize]
                };
                (
                    Vertex::Block(raw_target),
                    Some(PhiCopies {
                        from: source,
                        block: raw_target,
                    }),
                )
            }
            Vertex::Aux(aux) => {
                let fan_out =
                    &self.overlay.aux_vertices[aux.0 as usize].fan_out[arc.index as usize];
                (fan_out.target, None)
            }
            Vertex::Loop(scc) => {
                let loop_overlay = self.overlay.loops[scc.0 as usize]
                    .as_ref()
                    .expect("collapsed loop traversed before its loop pass ran");
                let raw_target = match loop_overlay.exit_demux {
                    Some(demux) => Vertex::Aux(demux),
                    None => {
                        debug_assert_eq!(arc.index, 0);
                        loop_overlay.exit_targets[0]
                    }
                };
                (raw_target, None)
            }
        };

        let (assignments, target): (&'a [AuxAssign], Vertex) = match self.overlay.rewrite_of(arc) {
            Some(rewrite) => (&rewrite.assignments, rewrite.redirect),
            None => (&[], self.present_collapsed(raw_target)),
        };
        TraversedArc {
            arc,
            phi_copies,
            assignments,
            target,
        }
    }

    /// Present a raw target at this view's nesting level: a block inside a
    /// processed loop appears as the collapsed loop vertex. Rewritten
    /// redirects are already level-correct and never pass through here.
    fn present_collapsed(&self, target: Vertex) -> Vertex {
        match target {
            Vertex::Block(block) => match self.collapsed_loop_of(block) {
                Some(scc) => Vertex::Loop(scc),
                None => target,
            },
            other => other,
        }
    }

    /// The processed loop `block` is collapsed into at this level, if any.
    /// A loop whose overlay record does not exist yet (the loop pass has
    /// not reached it) is not presented collapsed.
    fn collapsed_loop_of(&self, block: BasicBlockId) -> Option<SccTreeNodeId> {
        let scc = self.collapse.get(block.0 as usize).copied().flatten()?;
        self.overlay.loops[scc.0 as usize].as_ref().map(|_| scc)
    }

    /// Whether `arc` is a repetition arc hidden by this loop-body view: the
    /// loop tail's repeat alternative (restructured loop) or the original
    /// back edge (structured loop).
    fn is_hidden_repetition_arc(&self, arc: ArcId) -> bool {
        let Some(scc) = self.body_of else {
            return false;
        };
        let Some(loop_overlay) = self.overlay.loops[scc.0 as usize].as_ref() else {
            return false;
        };
        if loop_overlay.structured_back_edge == Some(arc) {
            return true;
        }
        match loop_overlay.tail {
            // The repeat alternative of the loop tail is fan-out index 1
            // (index 0 exits); see AuxVertexKind::LoopTail.
            Some(tail) => arc.source == Vertex::Aux(tail) && arc.index == 1,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::control_flow::overlay::{
        AuxAssign, AuxVar, AuxVertexId, AuxVertexKind, LoopOverlay,
    };
    use llvm_ir::Name;
    use smallvec::smallvec;

    fn block(id: u32) -> Vertex {
        Vertex::Block(BasicBlockId(id))
    }

    fn aux(id: AuxVertexId) -> Vertex {
        Vertex::Aux(id)
    }

    fn arc_from_block(source: u32, index: u32) -> ArcId {
        ArcId {
            source: block(source),
            index,
        }
    }

    /// The universal (root) view with no collapsing and no body hiding.
    fn root_view<'a>(
        mapper: &'a BasicBlockMapper,
        overlay: &'a Overlay,
        collapse: &'a [Option<SccTreeNodeId>],
    ) -> RegionView<'a> {
        RegionView {
            mapper,
            overlay,
            members: Membership::Universal,
            collapse,
            body_of: None,
        }
    }

    #[test]
    fn straight_line_traversal_carries_phi_copies() {
        // 0 -> 1 -> exit (via ret arc).
        let (mut mapper, exit) = {
            let mut mapper = BasicBlockMapper::new(3);
            for i in 0..2 {
                mapper.intern(&Name::Number(i));
            }
            let exit = mapper.intern(&mapper.exit_name());
            (mapper, exit)
        };
        mapper.add_connection(BasicBlockId(0), BasicBlockId(1));
        mapper.add_connection(BasicBlockId(1), exit);

        let overlay = Overlay::new(&mapper, vec![false; 3], 0);
        let collapse = vec![None; 3];
        let view = root_view(&mapper, &overlay, &collapse);

        let arcs = view.arcs_out(block(0));
        assert_eq!(1, arcs.len());
        assert_eq!(block(1), arcs[0].target);
        assert_eq!(
            Some(PhiCopies {
                from: BasicBlockId(0),
                block: BasicBlockId(1),
            }),
            arcs[0].phi_copies
        );
        assert!(arcs[0].assignments.is_empty());

        let ret_arcs = view.arcs_out(block(1));
        assert_eq!(1, ret_arcs.len());
        assert_eq!(Vertex::Block(exit), ret_arcs[0].target);
    }

    #[test]
    fn rewrite_redirects_and_updates_in_arc_lists() {
        // Branch 0 -> {1, 2}; both fan into 3.
        let (mut mapper, _exit) = {
            let mut mapper = BasicBlockMapper::new(5);
            for i in 0..4 {
                mapper.intern(&Name::Number(i));
            }
            let exit = mapper.intern(&mapper.exit_name());
            (mapper, exit)
        };
        mapper.add_connection(BasicBlockId(0), BasicBlockId(1));
        mapper.add_connection(BasicBlockId(0), BasicBlockId(2));
        mapper.add_connection(BasicBlockId(1), BasicBlockId(3));
        mapper.add_connection(BasicBlockId(2), BasicBlockId(3));

        let mut overlay = Overlay::new(&mapper, vec![false; 5], 0);
        let demux = overlay.add_aux_vertex(AuxVertexKind::BranchDemux, &[block(3)], None);
        let selector = AuxVar::ContinuationSelector(demux);
        overlay.rewrite_arc(
            arc_from_block(1, 0),
            &[AuxAssign {
                var: selector,
                value: 0,
            }],
            Vertex::Aux(demux),
        );

        let collapse = vec![None; 5];
        let view = root_view(&mapper, &overlay, &collapse);

        // The rewritten arc: original phi copies preserved, redirected.
        let arcs = view.arcs_out(block(1));
        assert_eq!(1, arcs.len());
        assert_eq!(Vertex::Aux(demux), arcs[0].target);
        assert_eq!(
            Some(PhiCopies {
                from: BasicBlockId(1),
                block: BasicBlockId(3),
            }),
            arcs[0].phi_copies
        );
        assert_eq!(1, arcs[0].assignments.len());

        // In-arc bookkeeping: the demux gained the arc, block 3 keeps only
        // the unrewritten arc from 2 plus the demux fan-out arc.
        let demux_in = view.arcs_in(Vertex::Aux(demux));
        assert_eq!(1, demux_in.len());
        assert_eq!(arc_from_block(1, 0), demux_in[0].arc);

        let join_in = view.arcs_in(block(3));
        assert_eq!(2, join_in.len());
        assert!(join_in.iter().any(|a| a.arc == arc_from_block(2, 0)));
        assert!(
            join_in
                .iter()
                .any(|a| a.arc.source == Vertex::Aux(demux) && a.phi_copies.is_none())
        );
    }

    #[test]
    fn appending_to_a_rewrite_fuses_assignments_in_order() {
        let (mut mapper, _exit) = {
            let mut mapper = BasicBlockMapper::new(3);
            for i in 0..2 {
                mapper.intern(&Name::Number(i));
            }
            let exit = mapper.intern(&mapper.exit_name());
            (mapper, exit)
        };
        mapper.add_connection(BasicBlockId(0), BasicBlockId(1));

        let scc = SccTreeNodeId(0);
        let mut overlay = Overlay::new(&mapper, vec![false; 3], 1);
        let first = AuxAssign {
            var: AuxVar::LoopVertexSelector(scc),
            value: 1,
        };
        overlay.rewrite_arc(arc_from_block(0, 0), &[first], Vertex::Loop(scc));

        // A later branch demux funnels the same arc: the continuation
        // selector fuses AFTER the loop selector, and the redirect moves.
        let demux = overlay.add_aux_vertex(AuxVertexKind::BranchDemux, &[Vertex::Loop(scc)], None);
        let second = AuxAssign {
            var: AuxVar::ContinuationSelector(demux),
            value: 0,
        };
        overlay.rewrite_arc(arc_from_block(0, 0), &[second], Vertex::Aux(demux));

        let rewrite = overlay.rewrite_of(arc_from_block(0, 0)).unwrap();
        assert_eq!(vec![first, second], rewrite.assignments.to_vec());
        assert_eq!(Vertex::Aux(demux), rewrite.redirect);

        // The loop vertex's overlay in-arcs no longer include the arc (it
        // moved to the demux); the demux fan-out arc enters it instead.
        assert!(
            !overlay
                .overlay_in_arcs(Vertex::Loop(scc))
                .contains(&arc_from_block(0, 0))
        );
        assert!(
            overlay
                .overlay_in_arcs(Vertex::Loop(scc))
                .iter()
                .any(|a| a.source == Vertex::Aux(demux))
        );
    }

    #[test]
    fn body_view_hides_repetition_arcs_and_exposes_back_edge_payload() {
        // Structured do-while: 0 -> 1(header); 1 -> 1 (back edge, arc 0),
        // 1 -> 2 (exit arc, arc 1); 2 -> exit.
        let (mut mapper, exit) = {
            let mut mapper = BasicBlockMapper::new(4);
            for i in 0..3 {
                mapper.intern(&Name::Number(i));
            }
            let exit = mapper.intern(&mapper.exit_name());
            (mapper, exit)
        };
        mapper.add_connection(BasicBlockId(0), BasicBlockId(1));
        mapper.add_connection(BasicBlockId(1), BasicBlockId(1));
        mapper.add_connection(BasicBlockId(1), BasicBlockId(2));
        mapper.add_connection(BasicBlockId(2), exit);

        let scc = SccTreeNodeId(0);
        let mut overlay = Overlay::new(&mapper, vec![false; 4], 1);
        overlay.loops[0] = Some(LoopOverlay {
            entries: smallvec![BasicBlockId(1)],
            exit_targets: smallvec![block(2)],
            entry_demux: None,
            tail: None,
            exit_demux: None,
            structured_back_edge: Some(arc_from_block(1, 0)),
        });
        // The entry arc still gets a redirect record: collapsing needs it.
        overlay.rewrite_arc(arc_from_block(0, 0), &[], Vertex::Loop(scc));

        let collapse: Vec<Option<SccTreeNodeId>> = vec![None, Some(scc), None, None];
        let body_collapse = vec![None; 4];

        // Parent view: 0's arc enters the collapsed loop; the loop's
        // successor arc is payload-free and reaches 2.
        let parent = root_view(&mapper, &overlay, &collapse);
        let entry_arcs = parent.arcs_out(block(0));
        assert_eq!(Vertex::Loop(scc), entry_arcs[0].target);
        let loop_arcs = parent.arcs_out(Vertex::Loop(scc));
        assert_eq!(1, loop_arcs.len());
        assert_eq!(block(2), loop_arcs[0].target);
        assert!(loop_arcs[0].phi_copies.is_none());

        // Parent arcs_in(2): the structured loop's exit arc presents as the
        // collapsed vertex's successor arc, not as an arc from block 1.
        let join_in = parent.arcs_in(block(2));
        assert_eq!(1, join_in.len());
        assert_eq!(Vertex::Loop(scc), join_in[0].arc.source);

        // Body view: the back edge is hidden, only the exit arc remains,
        // and the hidden back edge's payload is exposed for the emitter.
        let body = RegionView {
            mapper: &mapper,
            overlay: &overlay,
            members: Membership::Universal,
            collapse: &body_collapse,
            body_of: Some(scc),
        };
        let tail_arcs = body.arcs_out(block(1));
        assert_eq!(1, tail_arcs.len());
        assert_eq!(block(2), tail_arcs[0].target);

        let back_edge = body.hidden_back_edge().unwrap();
        assert_eq!(
            Some(PhiCopies {
                from: BasicBlockId(1),
                block: BasicBlockId(1),
            }),
            back_edge.phi_copies
        );

        // Inside the body the header is the region entry and has no
        // in-arcs: the rewritten entry arc enters the collapsed Loop vertex
        // (not the header block), and the back edge is hidden.
        assert!(body.arcs_in(block(1)).is_empty());
        assert!(
            overlay
                .overlay_in_arcs(Vertex::Loop(scc))
                .contains(&arc_from_block(0, 0))
        );
    }

    #[test]
    fn diverging_block_gets_synthetic_exit_arc() {
        // 0 branches to 1 (ret) and 2 (unreachable).
        let (mut mapper, exit) = {
            let mut mapper = BasicBlockMapper::new(4);
            for i in 0..3 {
                mapper.intern(&Name::Number(i));
            }
            let exit = mapper.intern(&mapper.exit_name());
            (mapper, exit)
        };
        mapper.add_connection(BasicBlockId(0), BasicBlockId(1));
        mapper.add_connection(BasicBlockId(0), BasicBlockId(2));
        mapper.add_connection(BasicBlockId(1), exit);

        let overlay = Overlay::new(&mapper, vec![false, false, true, false], 0);
        let collapse = vec![None; 4];
        let view = root_view(&mapper, &overlay, &collapse);

        let arcs = view.arcs_out(block(2));
        assert_eq!(1, arcs.len());
        assert_eq!(Vertex::Block(exit), arcs[0].target);
        assert_eq!(
            Some(PhiCopies {
                from: BasicBlockId(2),
                block: exit,
            }),
            arcs[0].phi_copies
        );

        // The exit block's in-arcs include both the real ret arc and the
        // synthetic diverging arc.
        let exit_in = view.arcs_in(Vertex::Block(exit));
        assert_eq!(2, exit_in.len());
    }

    #[test]
    fn restructured_loop_tail_repeat_alternative_is_hidden_in_body() {
        let (mapper, _exit) = {
            let mut mapper = BasicBlockMapper::new(2);
            mapper.intern(&Name::Number(0));
            let exit = mapper.intern(&mapper.exit_name());
            (mapper, exit)
        };

        let scc = SccTreeNodeId(0);
        let mut overlay = Overlay::new(&mapper, vec![false; 2], 1);
        // Tail fan-out: alternative 0 exits, alternative 1 repeats.
        let tail =
            overlay.add_aux_vertex(AuxVertexKind::LoopTail, &[block(0), block(0)], Some(scc));
        overlay.loops[0] = Some(LoopOverlay {
            entries: smallvec![BasicBlockId(0)],
            exit_targets: smallvec![block(0)],
            entry_demux: None,
            tail: Some(tail),
            exit_demux: None,
            structured_back_edge: None,
        });

        let collapse = vec![None; 2];
        let parent = root_view(&mapper, &overlay, &collapse);
        assert_eq!(2, parent.arcs_out(Vertex::Aux(tail)).len());

        let body = RegionView {
            mapper: &mapper,
            overlay: &overlay,
            members: Membership::Universal,
            collapse: &collapse,
            body_of: Some(scc),
        };
        let visible = body.arcs_out(Vertex::Aux(tail));
        assert_eq!(1, visible.len());
        assert_eq!(0, visible[0].arc.index, "only the exit alternative remains");
        assert!(body.hidden_back_edge().is_none());
    }

    #[test]
    fn vertex_set_stamps_and_clears() {
        let mut set = VertexSet::new(3, 1);
        set.insert(block(1));
        set.insert(Vertex::Loop(SccTreeNodeId(0)));
        assert!(set.contains(block(1)));
        assert!(!set.contains(block(0)));
        assert!(set.contains(Vertex::Loop(SccTreeNodeId(0))));
        // Aux slots grow on demand and read as absent until inserted.
        assert!(!set.contains(aux(AuxVertexId(5))));
        set.insert(aux(AuxVertexId(5)));
        assert!(set.contains(aux(AuxVertexId(5))));

        set.clear();
        assert!(!set.contains(block(1)));
        assert!(!set.contains(aux(AuxVertexId(5))));
    }
}
