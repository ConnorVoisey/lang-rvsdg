//! **Branch partitioning**: given a branch vertex in an acyclic region,
//! compute each alternative's subgraph (its "arm": the vertices reachable
//! only through that alternative and no other) and the continuation points
//! (where the arms meet again, or leave the region). This is the paper's
//! dominator graph of each fan-out arc (Bahmann et al. 2015, Def 2.5)
//! evaluated as a worklist: a vertex joins an arm exactly when every one of
//! its in-region predecessor arcs comes from an arm member, or is the
//! alternative's own fan-out arc.
//!
//! Shared by the branch restructuring pass (which inserts demuxes when a
//! partition has several continuation points) and the RVSDG emitter (which
//! re-partitions the restructured graph, where every branch has exactly one
//! continuation, to recover arm member sets for its recursion).
//!
//! Preconditions (hard): the region is acyclic (vertices on a cycle would
//! each wait for the other and silently fall out of every arm -- the loop
//! pass guarantees this) and every region vertex is reachable from the
//! region entry (unreachable blocks are pruned at intern time).

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::llvm_parser::control_flow::{
    overlay::{ArcId, Vertex},
    view::RegionView,
};

/// One fan-out alternative of the branch being partitioned: its arc and the
/// arc's current effective target.
#[derive(Clone, Copy, Debug)]
pub(in crate::llvm_parser) struct SeedArc {
    pub arc: ArcId,
    pub target: Vertex,
}

/// One grown arm. An empty member list means the alternative jumps straight
/// to a continuation point.
#[derive(Debug)]
pub(in crate::llvm_parser) struct Arm {
    pub seed: SeedArc,
    pub members: Vec<Vertex>,
}

/// The result of partitioning one branch.
#[derive(Debug)]
pub(in crate::llvm_parser) struct Partition {
    pub arms: Vec<Arm>,
    /// The continuation points in ascending vertex order (the selector
    /// numbering convention).
    pub continuations: SmallVec<[Vertex; 4]>,
    /// Union of every arm's members, for "is this vertex inside any arm"
    /// queries (the trimming rule and boundary-arc collection need them).
    pub arm_members: FxHashSet<Vertex>,
}

/// Reusable partitioning state: the per-vertex predecessor-counting slots.
/// One instance serves a whole pass; generations keep partitions apart
/// without clearing.
#[derive(Debug)]
pub(in crate::llvm_parser) struct Partitioner {
    blocks: Vec<GrowthSlot>,
    aux: Vec<GrowthSlot>,
    loops: Vec<GrowthSlot>,
    generation: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct GrowthSlot {
    stamp: u32,
    count: u32,
    indegree: u32,
}

impl Partitioner {
    pub(in crate::llvm_parser) fn new(block_count: usize, loop_count: usize) -> Self {
        Self {
            blocks: vec![GrowthSlot::default(); block_count],
            aux: Vec::new(),
            loops: vec![GrowthSlot::default(); loop_count],
            generation: 0,
        }
    }

    /// Partition `branch` (whose fan-out is `arcs`) within `view`'s region.
    pub(in crate::llvm_parser) fn partition(
        &mut self,
        view: &RegionView<'_>,
        branch: Vertex,
        arcs: &[SeedArc],
    ) -> Partition {
        let mut arms: Vec<Arm> = Vec::with_capacity(arcs.len());
        let mut arm_members: FxHashSet<Vertex> = FxHashSet::default();
        for &seed in arcs {
            self.generation += 1;
            let generation = self.generation;
            let mut members: Vec<Vertex> = Vec::new();
            let mut worklist: SmallVec<[Vertex; 8]> = SmallVec::new();

            self.count_arc(view, branch, generation, seed.target, &mut worklist, &mut members);
            while let Some(vertex) = worklist.pop() {
                for out in view.arcs_out(vertex) {
                    self.count_arc(view, branch, generation, out.target, &mut worklist, &mut members);
                }
            }
            arm_members.extend(members.iter().copied());
            arms.push(Arm { seed, members });
        }

        let continuations = continuation_points(view, branch, &arms, &arm_members);
        Partition {
            arms,
            continuations,
            arm_members,
        }
    }

    /// Count one arc into `target` for the arm currently being grown; the
    /// target joins (and is queued) once every in-region predecessor arc is
    /// accounted for.
    fn count_arc(
        &mut self,
        view: &RegionView<'_>,
        branch: Vertex,
        generation: u32,
        target: Vertex,
        worklist: &mut SmallVec<[Vertex; 8]>,
        members: &mut Vec<Vertex>,
    ) {
        if target == branch || !view.contains(target) {
            return;
        }
        // The indegree scan is O(in-arcs); do it only on the first touch of
        // this (arm, target) pair.
        let needs_indegree = self.slot(target).stamp != generation;
        let indegree = if needs_indegree {
            view.arcs_in(target)
                .iter()
                .filter(|incoming| view.contains(incoming.arc.source))
                .count() as u32
        } else {
            0
        };
        let slot = self.slot_mut(target);
        if slot.stamp != generation {
            slot.stamp = generation;
            slot.count = 0;
            slot.indegree = indegree;
        }
        slot.count += 1;
        debug_assert!(slot.count <= slot.indegree, "arc counted twice");
        if slot.count == slot.indegree {
            members.push(target);
            worklist.push(target);
        }
    }

    fn slot(&mut self, vertex: Vertex) -> GrowthSlot {
        *self.slot_mut(vertex)
    }

    fn slot_mut(&mut self, vertex: Vertex) -> &mut GrowthSlot {
        match vertex {
            Vertex::Block(b) => &mut self.blocks[b.0 as usize],
            Vertex::Aux(a) => {
                let index = a.0 as usize;
                if index >= self.aux.len() {
                    self.aux.resize(index + 1, GrowthSlot::default());
                }
                &mut self.aux[index]
            }
            Vertex::Loop(l) => &mut self.loops[l.0 as usize],
        }
    }
}

/// The continuation points of a partition: every non-arm target of an arm's
/// out-arc or of an empty alternative's fan-out arc, deduplicated and in
/// ascending vertex order. Re-queried from the view, so it stays correct
/// after the trimming rule re-points arcs (the branch pass calls this a
/// second time after promoting).
pub(in crate::llvm_parser) fn continuation_points(
    view: &RegionView<'_>,
    branch: Vertex,
    arms: &[Arm],
    arm_members: &FxHashSet<Vertex>,
) -> SmallVec<[Vertex; 4]> {
    let mut continuations: SmallVec<[Vertex; 4]> = SmallVec::new();
    let mut push = |vertex: Vertex| {
        if !continuations.contains(&vertex) {
            continuations.push(vertex);
        }
    };
    let fan_out = view.arcs_out(branch);
    for (arm, out) in arms.iter().zip(fan_out.iter()) {
        if arm.members.is_empty() {
            push(out.target);
        }
    }
    for arm in arms {
        for &member in &arm.members {
            for out in view.arcs_out(member) {
                if !arm_members.contains(&out.target) {
                    push(out.target);
                }
            }
        }
    }
    continuations.sort_unstable();
    continuations
}
