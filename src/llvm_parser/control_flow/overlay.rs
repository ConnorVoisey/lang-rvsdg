//! **The restructuring overlay**: sidecar records describing the virtually
//! restructured control flow graph, without ever copying or mutating the
//! interned CFG.
//!
//! The construction algorithm (Bahmann, Reissmann, Jahre, Meyer 2015,
//! section 4) restructures a function's control flow by inserting branch
//! vertices that route control according to small integer selector
//! variables, and by replacing arcs with "assign a selector constant, then
//! proceed to one of those vertices". Every such edit is representable as
//! one of two records:
//!
//! - an [`AuxVertex`]: a vertex the restructuring inserted (a demux branch
//!   or a promoted assignment), with its own outgoing arcs;
//! - an [`ArcRewrite`]: an existing arc now carries constant assignments to
//!   auxiliary variables and enters a different vertex (`redirect`).
//!
//! "Demux" throughout means a branch that routes control to one of several
//! destinations according to a selector value computed earlier on the path.
//!
//! The overlay composes with the immutable [`BasicBlockMapper`] through
//! [`super::view::RegionView`], which is what the restructuring passes and
//! the RVSDG emitter traverse. The overlay is a complete description of the
//! restructured graph: tests can expand it into an explicit CFG and check,
//! by the paper's short-circuiting argument (Corollary 5.7), that the
//! restructuring is faithful.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    scc::SccTreeNodeId,
};

/// Index into [`Overlay::aux_vertices`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::llvm_parser) struct AuxVertexId(pub u32);

/// Index into [`Overlay::rewrites`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::llvm_parser) struct ArcRewriteId(pub u32);

/// A vertex of the (virtually) restructured CFG. `Block` is an original LLVM
/// basic block. `Aux` is a vertex the restructuring inserted. `Loop` is a
/// strongly connected component collapsed to a single vertex: once a loop is
/// restructured, the enclosing graph treats the whole loop as one vertex
/// (paper section 4.1: "we treat all L* subgraphs as if each were a single
/// vertex").
///
/// The derived order (`Block` < `Aux` < `Loop`, then by id) is load-bearing:
/// continuation points are sorted by it and selector values are assigned
/// from the sorted order, so construction is deterministic across runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::llvm_parser) enum Vertex {
    Block(BasicBlockId),
    Aux(AuxVertexId),
    Loop(SccTreeNodeId),
}

/// An arc, identified by its source vertex and the index of the outgoing
/// alternative: index 0 for unconditional arcs; for a conditional branch,
/// 0 = true and 1 = false; for a switch, 0 = default then the cases in
/// declaration order (the same order the block mapper stores `outputs` in).
///
/// Identifying arcs by (source, index) rather than (source, target) keeps
/// the graph a multigraph: two switch cases jumping to the same block are
/// distinct arcs, which matters because a branch alternative's subgraph is
/// per-arc (a block targeted by two alternatives of the same switch belongs
/// to neither alternative's subgraph).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::llvm_parser) struct ArcId {
    pub source: Vertex,
    pub index: u32,
}

/// One auxiliary variable invented by restructuring.
///
/// Selector symbols are INTEGER-typed values, not RVSDG predicates. RVSDG
/// predicate types carry their alternative count, and one selector can feed
/// demuxes with different counts (the loop vertex selector feeds an entry
/// demux with one count and an exit demux with another), so the symbol
/// itself stays an integer and each demux emits its own match conversion to
/// its own predicate type. The match node is then the predicate-defining
/// node: single-use and adjacent to its consumer, which is what predicate
/// continuation form requires; the paper explicitly blesses such conversion
/// nodes (the remark under its Definition 2.6: conversion nodes between
/// predicates and other kinds of values may be inserted freely).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::llvm_parser) enum AuxVar {
    /// Which vertex of a loop control is headed for (the paper's q). On
    /// loop entry and on each repeat it holds the index of the entry vertex
    /// to resume at; on loop exit it holds the index of the exit target
    /// that was chosen. The two uses never overlap: every path assigns it
    /// immediately before the demux that reads it.
    LoopVertexSelector(SccTreeNodeId),
    /// Whether the loop repeats (the paper's r): 1 to run the body again,
    /// 0 to leave the loop. Becomes the theta node's repetition predicate.
    LoopRepeat(SccTreeNodeId),
    /// Which continuation point a branch demux routes to (the paper's p).
    ContinuationSelector(AuxVertexId),
}

/// A constant assignment to an auxiliary variable. Restructuring never
/// assigns anything else (the paper only ever inserts q := k, r := 0 or 1,
/// p := k).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::llvm_parser) struct AuxAssign {
    pub var: AuxVar,
    pub value: u32,
}

/// A vertex inserted by restructuring. The demux kinds are branch
/// statements on an auxiliary variable. `PromotedAssign` is an assignment
/// vertex created only by the trimming rule of branch restructuring; it is
/// straight-line with a single successor.
#[derive(Debug)]
pub(in crate::llvm_parser) enum AuxVertexKind {
    /// Routes control to the right entry vertex of a multi-entry loop, on
    /// entry and on each repeat. Branches on the loop's LoopVertexSelector.
    LoopEntryDemux { scc: SccTreeNodeId },
    /// Decides at the end of each iteration whether the loop repeats.
    /// Branches on LoopRepeat: alternative 1 repeats (to the entry demux or
    /// the single entry vertex), alternative 0 leaves (to the exit demux or
    /// the single exit target). Its loop is recoverable through
    /// `LoopOverlay::tail`, so the variant carries no field.
    LoopTail,
    /// Routes control, after the loop, to the exit target that was chosen
    /// inside it. Branches on the loop's LoopVertexSelector.
    LoopExitDemux { scc: SccTreeNodeId },
    /// Routes control to the continuation point selected inside a branch's
    /// alternatives. Branches on its own ContinuationSelector.
    BranchDemux,
    /// An auxiliary assignment cluster lifted off an arc into a real vertex
    /// by the trimming rule, so that a later demux insertion cannot come
    /// between the assignments and the branch that reads them.
    PromotedAssign {
        assignments: SmallVec<[AuxAssign; 3]>,
    },
}

/// One outgoing arc of an inserted vertex. Fan-out arcs are first-class
/// arcs: a later restructuring step may rewrite one exactly like an
/// original arc. This happens whenever restructured constructs chain: when
/// loop A's exit target is inside loop B, B's entry-selector assignment
/// must ride A's exit-demux fan-out arc (the arc that now enters B),
/// because the original arc already carries A's exit assignments and those
/// belong to A's construct.
#[derive(Clone, Copy, Debug)]
pub(in crate::llvm_parser) struct FanOutArc {
    pub target: Vertex,
    pub rewrite: Option<ArcRewriteId>,
}

/// A vertex inserted by restructuring, with its outgoing arcs in
/// alternative order. For demux kinds the arc index is the selector value
/// that chooses it. `PromotedAssign` has exactly one outgoing arc.
#[derive(Debug)]
pub(in crate::llvm_parser) struct AuxVertex {
    pub kind: AuxVertexKind,
    pub fan_out: SmallVec<[FanOutArc; 4]>,
}

/// The rewrite of one arc. The paper replaces an arc with "an assignment
/// that proceeds to a demux vertex" (sections 4.1 and 4.2); here the
/// assignments ride the arc record and `redirect` is the vertex the arc now
/// enters. The inline capacity is 3 because a loop exit arc can carry the
/// loop's selector and repeat assignments and later gain a continuation
/// selector from a branch demux inside the same loop body.
#[derive(Debug)]
pub(in crate::llvm_parser) struct ArcRewrite {
    pub assignments: SmallVec<[AuxAssign; 3]>,
    pub redirect: Vertex,
}

/// The restructuring of one strongly connected component. Written by the
/// loop pass; read by the region view (collapsing) and the theta emitter.
#[derive(Debug)]
pub(in crate::llvm_parser) struct LoopOverlay {
    /// Entry vertices in ascending block order; the index is the
    /// LoopVertexSelector value that resumes at that entry.
    pub entries: SmallVec<[BasicBlockId; 2]>,
    /// Distinct exit-arc targets in ascending vertex order; the index is
    /// the LoopVertexSelector value assigned when leaving toward that
    /// target. Empty for an endless loop: the collapsed vertex then has no
    /// successor and the enclosing walk ends there.
    pub exit_targets: SmallVec<[Vertex; 2]>,
    /// The entry demux; None when there is a single entry vertex (a demux
    /// with one destination routes nothing).
    pub entry_demux: Option<AuxVertexId>,
    /// The loop tail; None when the loop is already structured: a single
    /// entry vertex and a single vertex carrying both the only repetition
    /// arc and the only exit arc. A structured loop gets no auxiliary
    /// variables at all; its own tail branch condition becomes the theta
    /// repetition predicate directly.
    pub tail: Option<AuxVertexId>,
    /// The exit demux; None when there are fewer than two exit targets.
    pub exit_demux: Option<AuxVertexId>,
    /// For a structured loop (`tail` is None): its single repetition arc.
    /// The loop body view hides this arc for control flow (the body must be
    /// acyclic), but its phi copies still apply at the body's end -- they
    /// define the loop variables' next-iteration values.
    pub structured_back_edge: Option<ArcId>,
}

/// Per-vertex storage for the three dense vertex id spaces. The aux space
/// grows as restructuring inserts vertices; the other two are fixed per
/// function.
#[derive(Debug, Default)]
struct PerVertex<T> {
    blocks: Vec<T>,
    aux: Vec<T>,
    loops: Vec<T>,
}

impl<T: Default + Clone> PerVertex<T> {
    fn new(block_count: usize, loop_count: usize) -> Self {
        Self {
            blocks: vec![T::default(); block_count],
            aux: Vec::new(),
            loops: vec![T::default(); loop_count],
        }
    }

    fn get(&self, vertex: Vertex) -> &T {
        match vertex {
            Vertex::Block(b) => &self.blocks[b.0 as usize],
            Vertex::Aux(a) => &self.aux[a.0 as usize],
            Vertex::Loop(l) => &self.loops[l.0 as usize],
        }
    }

    fn get_mut(&mut self, vertex: Vertex) -> &mut T {
        match vertex {
            Vertex::Block(b) => &mut self.blocks[b.0 as usize],
            Vertex::Aux(a) => &mut self.aux[a.0 as usize],
            Vertex::Loop(l) => &mut self.loops[l.0 as usize],
        }
    }

    fn push_aux(&mut self) {
        self.aux.push(T::default());
    }
}

/// The overlay for one function: every record the restructuring passes
/// write, plus the per-function graph facts traversal needs (the reverse
/// arc index and the diverging-block markers).
#[derive(Debug)]
pub(in crate::llvm_parser) struct Overlay {
    /// The synthetic exit block (interned by the block mapper; `ret` arcs
    /// to it are real arcs, diverging blocks get a synthetic one, below).
    pub exit_block: BasicBlockId,
    pub aux_vertices: Vec<AuxVertex>,
    /// Parallel to `aux_vertices`: the loop whose body each inserted vertex
    /// sits inside (None for the function root region).
    pub aux_owners: Vec<Option<SccTreeNodeId>>,
    pub rewrites: Vec<ArcRewrite>,
    /// Rewrite lookup for original (block-source) arcs: indexed exactly
    /// like the block mapper's per-block `outputs`, so the lookup is a
    /// direct load, no hashing (the emitter traverses arcs constantly).
    block_arc_rewrites: Vec<SmallVec<[Option<ArcRewriteId>; 2]>>,
    /// Rewrite lookup for overlay-created arcs whose source is a Loop
    /// vertex (a collapsed loop's single successor arc). Aux fan-out arcs
    /// store their rewrite inline in [`FanOutArc`] instead.
    loop_arc_rewrites: FxHashMap<ArcId, ArcRewriteId>,
    /// Per vertex: the arcs the OVERLAY routes into it -- rewritten arcs
    /// whose redirect enters it plus aux fan-out arcs targeting it.
    /// Original unrewritten arcs are found through `reverse` instead.
    overlay_in: PerVertex<SmallVec<[ArcId; 2]>>,
    /// Original in-arcs per block: (source block, arc index) pairs, in
    /// deterministic order. Includes the synthetic diverging-block arcs to
    /// the exit block. Built once; never changes (rewrites are filtered at
    /// query time by their records).
    reverse: Vec<SmallVec<[(BasicBlockId, u32); 4]>>,
    /// Blocks whose terminator is `unreachable` (or otherwise diverging):
    /// the interned graph gives them no successor, but the closed-CFG
    /// convention gives them a synthetic arc to the exit block whose "phi
    /// copy" binds the return value to poison.
    diverging: Vec<bool>,
    /// Per strongly connected component (indexed by tree node id): its
    /// restructuring, filled in by the loop pass.
    pub loops: Vec<Option<LoopOverlay>>,
}

impl Overlay {
    /// Set up an empty overlay over `mapper`. `diverging[b]` marks blocks
    /// whose terminator diverges (no successors); they get a synthetic arc
    /// to the exit block. `loop_count` is the SCC tree's node count.
    pub(in crate::llvm_parser) fn new(
        mapper: &BasicBlockMapper,
        diverging: Vec<bool>,
        loop_count: usize,
    ) -> Self {
        let block_count = mapper.blocks.len();
        debug_assert_eq!(diverging.len(), block_count);
        let exit_block = *mapper.get_exit_expect();

        let mut reverse: Vec<SmallVec<[(BasicBlockId, u32); 4]>> =
            vec![SmallVec::new(); block_count];
        let mut block_arc_rewrites: Vec<SmallVec<[Option<ArcRewriteId>; 2]>> =
            Vec::with_capacity(block_count);
        for source_index in 0..block_count {
            let source = BasicBlockId(source_index as u32);
            let outputs = mapper.outputs(source);
            for (arc_index, &target) in outputs.iter().enumerate() {
                reverse[target.0 as usize].push((source, arc_index as u32));
            }
            let mut slots: SmallVec<[Option<ArcRewriteId>; 2]> = SmallVec::new();
            // The synthetic diverging arc occupies index 0 of a block that
            // otherwise has no outputs; give it a rewrite slot too.
            let slot_count = outputs.len().max(usize::from(diverging[source_index]));
            slots.resize(slot_count, None);
            block_arc_rewrites.push(slots);
            if diverging[source_index] {
                debug_assert!(outputs.is_empty(), "diverging block with successors");
                reverse[exit_block.0 as usize].push((source, 0));
            }
        }

        Self {
            exit_block,
            aux_vertices: Vec::new(),
            aux_owners: Vec::new(),
            rewrites: Vec::new(),
            block_arc_rewrites,
            loop_arc_rewrites: FxHashMap::default(),
            overlay_in: PerVertex::new(block_count, loop_count),
            reverse,
            diverging,
            loops: (0..loop_count).map(|_| None).collect(),
        }
    }

    pub(in crate::llvm_parser) fn is_diverging(&self, block: BasicBlockId) -> bool {
        self.diverging[block.0 as usize]
    }

    /// Original in-arcs of `block` as (source block, arc index) pairs,
    /// including the synthetic diverging arcs into the exit block. Callers
    /// must filter arcs that have been rewritten away (the region view
    /// does).
    pub(in crate::llvm_parser) fn original_in_arcs(
        &self,
        block: BasicBlockId,
    ) -> &[(BasicBlockId, u32)] {
        &self.reverse[block.0 as usize]
    }

    /// Arcs the overlay currently routes into `vertex`: rewritten arcs
    /// redirected here plus aux fan-out arcs targeting it.
    pub(in crate::llvm_parser) fn overlay_in_arcs(&self, vertex: Vertex) -> &[ArcId] {
        self.overlay_in.get(vertex)
    }

    /// Insert an auxiliary vertex with the given fan-out targets. Each
    /// fan-out arc is registered as an overlay in-arc of its target --
    /// EXCEPT for a loop tail's: the tail is interior to its theta, its
    /// repeat alternative is hidden inside the body, and its exit
    /// alternative is represented at the parent level by the collapsed
    /// loop's successor arc. Registering them would give the loop's
    /// successor a phantom in-arc from an interior vertex that no
    /// partition's arm can ever own.
    /// `owner` is the loop whose body the vertex sits inside (None for the
    /// function root region); body member sets are built from it.
    pub(in crate::llvm_parser) fn add_aux_vertex(
        &mut self,
        kind: AuxVertexKind,
        fan_out_targets: &[Vertex],
        owner: Option<SccTreeNodeId>,
    ) -> AuxVertexId {
        let register_fan_out = !matches!(kind, AuxVertexKind::LoopTail);
        let id = AuxVertexId(self.aux_vertices.len() as u32);
        let fan_out = fan_out_targets
            .iter()
            .map(|&target| FanOutArc {
                target,
                rewrite: None,
            })
            .collect();
        self.aux_vertices.push(AuxVertex { kind, fan_out });
        self.aux_owners.push(owner);
        self.overlay_in.push_aux();
        if register_fan_out {
            for (index, &target) in fan_out_targets.iter().enumerate() {
                self.overlay_in.get_mut(target).push(ArcId {
                    source: Vertex::Aux(id),
                    index: index as u32,
                });
            }
        }
        id
    }

    /// The rewrite record of `arc`, if any.
    pub(in crate::llvm_parser) fn rewrite_of(&self, arc: ArcId) -> Option<&ArcRewrite> {
        self.rewrite_id_of(arc)
            .map(|id| &self.rewrites[id.0 as usize])
    }

    fn rewrite_id_of(&self, arc: ArcId) -> Option<ArcRewriteId> {
        match arc.source {
            Vertex::Block(b) => self.block_arc_rewrites[b.0 as usize]
                .get(arc.index as usize)
                .copied()
                .flatten(),
            Vertex::Aux(a) => self.aux_vertices[a.0 as usize].fan_out[arc.index as usize].rewrite,
            Vertex::Loop(_) => self.loop_arc_rewrites.get(&arc).copied(),
        }
    }

    /// Rewrite `arc`: append `assignments` to its cluster and redirect it
    /// to `redirect`. Creating or re-redirecting a rewrite keeps the
    /// per-vertex overlay in-arc lists in sync (the arc leaves its previous
    /// redirect target's list and joins the new one).
    ///
    /// Appending preserves evaluation order: assignments written by an
    /// earlier construct run before ones appended later, which is the
    /// paper's fusion rule (a continuation-selector assignment fusing after
    /// a loop-selector assignment on the same arc, section 4.2).
    pub(in crate::llvm_parser) fn rewrite_arc(
        &mut self,
        arc: ArcId,
        assignments: &[AuxAssign],
        redirect: Vertex,
    ) -> ArcRewriteId {
        match self.rewrite_id_of(arc) {
            Some(id) => {
                let previous = self.rewrites[id.0 as usize].redirect;
                if previous != redirect {
                    let list = self.overlay_in.get_mut(previous);
                    if let Some(position) = list.iter().position(|&a| a == arc) {
                        list.swap_remove(position);
                    }
                    self.overlay_in.get_mut(redirect).push(arc);
                    self.rewrites[id.0 as usize].redirect = redirect;
                }
                self.rewrites[id.0 as usize]
                    .assignments
                    .extend_from_slice(assignments);
                id
            }
            None => {
                // An overlay arc (aux fan-out, registered loop successor)
                // already sits in its current target's overlay in-arc list;
                // redirecting it for the first time must move it. Original
                // block arcs are found through the reverse index instead
                // and are filtered by their rewrite record, so they have no
                // list entry to remove.
                if let Some(previous) = self.overlay_arc_target(arc) {
                    let list = self.overlay_in.get_mut(previous);
                    if let Some(position) = list.iter().position(|&a| a == arc) {
                        list.swap_remove(position);
                    }
                }
                let id = ArcRewriteId(self.rewrites.len() as u32);
                self.rewrites.push(ArcRewrite {
                    assignments: assignments.iter().copied().collect(),
                    redirect,
                });
                match arc.source {
                    Vertex::Block(b) => {
                        self.block_arc_rewrites[b.0 as usize][arc.index as usize] = Some(id);
                    }
                    Vertex::Aux(a) => {
                        self.aux_vertices[a.0 as usize].fan_out[arc.index as usize].rewrite =
                            Some(id);
                    }
                    Vertex::Loop(_) => {
                        self.loop_arc_rewrites.insert(arc, id);
                    }
                }
                self.overlay_in.get_mut(redirect).push(arc);
                id
            }
        }
    }

    /// The current target of an overlay-created, not-yet-rewritten arc: an
    /// aux vertex's fan-out target, or a collapsed loop's successor. `None`
    /// for original block arcs.
    fn overlay_arc_target(&self, arc: ArcId) -> Option<Vertex> {
        match arc.source {
            Vertex::Block(_) => None,
            Vertex::Aux(a) => {
                Some(self.aux_vertices[a.0 as usize].fan_out[arc.index as usize].target)
            }
            Vertex::Loop(scc) => {
                let loop_overlay = self.loops[scc.0 as usize].as_ref()?;
                match loop_overlay.exit_demux {
                    Some(demux) => Some(Vertex::Aux(demux)),
                    None => loop_overlay.exit_targets.first().copied(),
                }
            }
        }
    }

    /// The trimming rule's promotion: lift `arc`'s assignment cluster into
    /// its own `PromotedAssign` vertex placed in front of the arc's current
    /// redirect, leaving the arc itself clean and pointing at the promoted
    /// vertex. A demux funnel inserted afterwards then lands on the clean
    /// arc, BEFORE the assignments, keeping the cluster adjacent to the
    /// construct that consumes it.
    pub(in crate::llvm_parser) fn promote_arc_assignments(
        &mut self,
        arc: ArcId,
        owner: Option<SccTreeNodeId>,
    ) -> AuxVertexId {
        let rewrite_id = self
            .rewrite_id_of(arc)
            .expect("only assignment-carrying (rewritten) arcs are promoted");
        let assignments =
            std::mem::take(&mut self.rewrites[rewrite_id.0 as usize].assignments);
        debug_assert!(!assignments.is_empty(), "promotion of an empty cluster");
        let continuation = self.rewrites[rewrite_id.0 as usize].redirect;
        let promoted = self.add_aux_vertex(
            AuxVertexKind::PromotedAssign { assignments },
            &[continuation],
            owner,
        );
        // Re-point the (now clean) arc at the promoted vertex; the existing
        // rewrite machinery moves the overlay in-arc bookkeeping.
        self.rewrite_arc(arc, &[], Vertex::Aux(promoted));
        promoted
    }

    /// Enumerate the members of `scc`'s body region: the component's blocks
    /// with direct children collapsed (their blocks excluded, one Loop
    /// vertex each), plus every aux vertex restructuring placed inside this
    /// body. `child_collapse` is the body's nesting-level table
    /// (`SccTree::collapse_table` over the component's direct children).
    ///
    /// This is the ONE definition of body membership, shared by the branch
    /// pass and the emitter. A missing member becomes a permanently
    /// unresolvable continuation point that makes the branch pass insert
    /// routing demuxes forever, so the two consumers must never drift.
    pub(in crate::llvm_parser) fn for_each_body_member(
        &self,
        tree: &crate::llvm_parser::scc::SccTree,
        scc: SccTreeNodeId,
        child_collapse: &[Option<SccTreeNodeId>],
        mut member: impl FnMut(Vertex),
    ) {
        for &block in &tree.blocks[scc.0 as usize] {
            if child_collapse[block.0 as usize].is_none() {
                member(Vertex::Block(block));
            }
        }
        for &child in &tree.children[scc.0 as usize] {
            member(Vertex::Loop(child));
        }
        for (index, owner) in self.aux_owners.iter().enumerate() {
            if *owner == Some(scc) {
                member(Vertex::Aux(AuxVertexId(index as u32)));
            }
        }
    }

    /// Register a restructured single-exit-target loop's successor arc
    /// (`Loop(scc)`, index 0) as an overlay in-arc of `target`, so later
    /// passes classifying `target`'s in-arcs see the collapsed loop enter
    /// it. A structured loop needs no registration (its unrewritten exit
    /// arc is presented as the successor arc by the view), and a multi-exit
    /// loop's demux fan-out arcs are registered when the demux is inserted.
    pub(in crate::llvm_parser) fn register_loop_successor(
        &mut self,
        scc: SccTreeNodeId,
        target: Vertex,
    ) {
        self.overlay_in.get_mut(target).push(ArcId {
            source: Vertex::Loop(scc),
            index: 0,
        });
    }
}
