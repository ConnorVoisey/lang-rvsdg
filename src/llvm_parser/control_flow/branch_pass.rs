//! **The branch restructuring pass** (Bahmann, Reissmann, Jahre, Meyer 2015,
//! section 4.2): after the loop pass has collapsed every strongly connected
//! component, each region (the function root, every loop body, and
//! recursively every branch alternative's subgraph) is acyclic. This pass
//! walks each region and, at every branch whose alternatives rejoin at more
//! than one place, inserts a branch demux: a routing vertex that a selector
//! value, assigned on the arcs leaving the alternatives, steers to the right
//! continuation. After this pass every branch in the virtually restructured
//! graph has exactly one continuation point, which is what makes the RVSDG
//! emission a mechanical walk.
//!
//! Terms (see also the overlay module):
//!
//! - the **alternatives' subgraphs (arms)**: for one branch alternative, the
//!   vertices reachable only through that alternative and no other. A vertex
//!   joins an arm exactly when every one of its in-region predecessor arcs
//!   comes from an arm member (or is the alternative's own fan-out arc),
//!   which is the paper's dominator graph of the fan-out arc evaluated as a
//!   worklist.
//! - a **continuation point**: a vertex where two or more arms meet again,
//!   or where an arm leaves the region. One continuation point means the
//!   branch is an ordinary split/join; several mean routing is needed.
//! - **trimming**: before inserting a demux, an auxiliary-assignment cluster
//!   riding an arm's arc into a continuation point is lifted into its own
//!   `PromotedAssign` vertex on the tail side when the continuation also has
//!   predecessors outside the arms. This keeps the cluster adjacent to the
//!   construct that consumes it -- the demux funnel lands on the arc BEFORE
//!   the assignments -- reproducing the paper's figure 5 (section 4.2.1).
//!   When every predecessor is an arm (the paper's section 4.2 fusion rule:
//!   a continuation-selector assignment immediately following another
//!   auxiliary assignment merges into one vertex), the selector is appended
//!   to the same cluster instead.
//!
//! The pass writes overlay records only; nothing is emitted here. Together
//! with the loop pass it forms `build_overlay`: a complete description of
//! the restructured control flow graph that the emitter then walks.

use smallvec::SmallVec;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    control_flow::{
        overlay::{ArcId, AuxAssign, AuxVar, AuxVertexId, AuxVertexKind, Overlay, Vertex},
        partition::{Partition, Partitioner, SeedArc, continuation_points},
        view::{Membership, RegionView, VertexSet},
    },
    scc::{SccTree, SccTreeNodeId},
};

/// Build the region view for `scope`. A free function over the pass's
/// fields (rather than a `&self` method) so a view can coexist with a
/// mutable borrow of the partitioner, which lives in a different field.
fn make_view<'v>(
    mapper: &'v BasicBlockMapper,
    overlay: &'v Overlay,
    set_pool: &'v [VertexSet],
    scope: &Scope<'v>,
) -> RegionView<'v> {
    RegionView {
        mapper,
        overlay,
        members: match scope.members {
            None => Membership::Universal,
            Some(index) => Membership::Stamps(&set_pool[index]),
        },
        collapse: scope.collapse,
        body_of: scope.body_of,
    }
}

/// Run the branch pass over the function root and, recursively, every loop
/// body and branch alternative. Requires the loop pass to have run (every
/// region is acyclic under the views this pass builds).
#[tracing::instrument(name = "branch_pass", skip_all, fields(blocks = mapper.blocks.len()))]
pub(in crate::llvm_parser) fn run_branch_pass(
    mapper: &BasicBlockMapper,
    tree: &SccTree,
    overlay: &mut Overlay,
) {
    let block_count = mapper.blocks.len();
    let root_collapse = tree.collapse_table(&tree.roots, block_count);
    let mut pass = BranchPass {
        mapper,
        tree,
        overlay,
        set_pool: Vec::new(),
        partitioner: Partitioner::new(block_count, tree.len()),
    };
    pass.walk_region(
        Vertex::Block(BasicBlockId(0)),
        &Scope {
            members: None,
            collapse: &root_collapse,
            body_of: None,
        },
        0,
    );
}

/// One region's scope: its member set (None for the universal root), the
/// collapse table of its nesting level, and the enclosing loop whose
/// repetition arcs are hidden.
struct Scope<'a> {
    /// Index into the set pool; None means the universal root region.
    members: Option<usize>,
    collapse: &'a [Option<SccTreeNodeId>],
    body_of: Option<SccTreeNodeId>,
}

struct BranchPass<'a> {
    mapper: &'a BasicBlockMapper,
    tree: &'a SccTree,
    overlay: &'a mut Overlay,
    /// Reusable member sets, one per recursion depth. An arm or loop body at
    /// depth d+1 clears and refills slot d+1; stack discipline guarantees a
    /// slot is never needed by two live regions at once.
    set_pool: Vec<VertexSet>,
    partitioner: Partitioner,
}

impl BranchPass<'_> {
    /// Walk one region from `entry`, restructuring every branch met. The
    /// tail after a branch continues THIS loop iteration (never a recursive
    /// call), so call depth is bounded by construct nesting, not by chain
    /// length.
    fn walk_region(&mut self, entry: Vertex, scope: &Scope<'_>, depth: usize) {
        let mut current = entry;
        // Regions are finite and acyclic, so the walk is bounded by the
        // vertex count plus the demuxes it inserts (at most one per
        // partition). Exceeding a generous multiple means a vertex escaped
        // some region's member set and became an unresolvable continuation:
        // fail loudly instead of inserting demuxes forever.
        let step_limit = 16 * (self.mapper.blocks.len() as u32 + 64);
        let mut steps = 0u32;
        loop {
            steps += 1;
            assert!(
                steps < step_limit,
                "branch pass walk failed to converge at {current:?} \
                 (entry {entry:?}, depth {depth}): a region member set is incomplete"
            );
            if let Vertex::Loop(scc) = current {
                self.descend_into_body(scc, depth);
            }

            let arcs = self.out_arcs(current, scope);
            match arcs.len() {
                0 => return,
                1 => {
                    if !self.is_member(arcs[0].target, scope) {
                        return;
                    }
                    current = arcs[0].target;
                }
                _ => {
                    let partition = {
                        let view = make_view(self.mapper, self.overlay, &self.set_pool, scope);
                        self.partitioner.partition(&view, current, &arcs)
                    };
                    // Zero continuation points is legitimate: every
                    // alternative ends the region within itself (reaches
                    // the exit block inside its own subgraph, or diverges
                    // into an endless loop). Nothing rejoins, so there is
                    // nothing to route.
                    let join = match partition.continuations.len() {
                        0 => None,
                        1 => Some(partition.continuations[0]),
                        _ => {
                            let demux = self.trim_and_insert_demux(current, &partition, scope);
                            Some(Vertex::Aux(demux))
                        }
                    };

                    for arm in &partition.arms {
                        if arm.members.is_empty() {
                            continue;
                        }
                        let arm_depth = depth + 1;
                        self.fill_pool_set(arm_depth, &arm.members);
                        let arm_scope = Scope {
                            members: Some(arm_depth),
                            collapse: scope.collapse,
                            body_of: scope.body_of,
                        };
                        self.walk_region(arm.seed.target, &arm_scope, arm_depth);
                    }

                    match join {
                        Some(join) if self.is_member(join, scope) => current = join,
                        _ => return,
                    }
                }
            }
        }
    }

    /// Restructure the branches inside one collapsed loop's body.
    fn descend_into_body(&mut self, scc: SccTreeNodeId, depth: usize) {
        let record = self.overlay.loops[scc.0 as usize]
            .as_ref()
            .expect("branch pass reached a loop the loop pass skipped");
        let entry = match record.entry_demux {
            Some(demux) => Vertex::Aux(demux),
            None => Vertex::Block(record.entries[0]),
        };

        // The body's nesting level: this component's direct children are
        // collapsed; their blocks are not body members.
        let child_collapse = self.tree.collapse_table(
            &self.tree.children[scc.0 as usize],
            self.mapper.blocks.len(),
        );

        let body_depth = depth + 1;
        self.ensure_pool(body_depth);
        let set = &mut self.set_pool[body_depth];
        set.clear();
        self.overlay
            .for_each_body_member(self.tree, scc, &child_collapse, |vertex| {
                set.insert(vertex);
            });

        let body_scope = Scope {
            members: Some(body_depth),
            collapse: &child_collapse,
            body_of: Some(scc),
        };
        self.walk_region(entry, &body_scope, body_depth);
    }

    /// Apply the trimming rule, then insert the branch demux and the
    /// selector rewrites. Returns the demux vertex, which becomes the
    /// branch's single continuation.
    fn trim_and_insert_demux(
        &mut self,
        branch: Vertex,
        partition: &Partition,
        scope: &Scope<'_>,
    ) -> AuxVertexId {
        // Trimming (paper section 4.2.1). For each continuation point that
        // follows an auxiliary assignment: unless every in-region
        // predecessor lies inside an arm, lift each assignment-carrying
        // arm arc's cluster into a PromotedAssign vertex on the tail side.
        // The demux funnel then lands on the (now clean) arc BEFORE the
        // assignments, keeping them adjacent to the construct that consumes
        // them. When every predecessor IS an arm, the selector is instead
        // appended to the cluster: the paper's fusion rule.
        for &continuation in &partition.continuations {
            let mut carrying_arm_arcs: SmallVec<[ArcId; 4]> = SmallVec::new();
            let mut any_aux_pred = false;
            let mut all_preds_in_arms = true;
            {
                let view = make_view(self.mapper, self.overlay, &self.set_pool, scope);
                for incoming in view.arcs_in(continuation) {
                    if !self.is_member(incoming.arc.source, scope) {
                        continue;
                    }
                    let from_promoted = matches!(
                        incoming.arc.source,
                        Vertex::Aux(a) if matches!(
                            self.overlay.aux_vertices[a.0 as usize].kind,
                            AuxVertexKind::PromotedAssign { .. }
                        )
                    );
                    if !incoming.assignments.is_empty() || from_promoted {
                        any_aux_pred = true;
                    }
                    if partition.arm_members.contains(&incoming.arc.source) {
                        if !incoming.assignments.is_empty() {
                            carrying_arm_arcs.push(incoming.arc);
                        }
                    } else {
                        all_preds_in_arms = false;
                    }
                }
            }
            if !any_aux_pred || all_preds_in_arms {
                continue;
            }
            for arc in carrying_arm_arcs {
                let promoted = self.overlay.promote_arc_assignments(arc, scope.body_of);
                self.add_region_member(Vertex::Aux(promoted), scope);
            }
        }

        // Promotion may have re-pointed arcs; recompute the continuations
        // from a fresh view.
        let continuations = {
            let view = make_view(self.mapper, self.overlay, &self.set_pool, scope);
            continuation_points(&view, branch, &partition.arms, &partition.arm_members)
        };
        debug_assert!(continuations.len() > 1);

        // The demux, and the selector rewrites funneling every arm-boundary
        // arc (and every empty alternative's fan-out arc) through it.
        let demux =
            self.overlay
                .add_aux_vertex(AuxVertexKind::BranchDemux, &continuations, scope.body_of);
        self.add_region_member(Vertex::Aux(demux), scope);
        let selector = AuxVar::ContinuationSelector(demux);
        let selector_value = |target: Vertex| {
            continuations
                .iter()
                .position(|&c| c == target)
                .expect("boundary target is a continuation point") as u32
        };

        let mut to_rewrite: SmallVec<[SeedArc; 8]> = SmallVec::new();
        {
            let view = make_view(self.mapper, self.overlay, &self.set_pool, scope);
            let fan_out = view.arcs_out(branch);
            for (arm, out) in partition.arms.iter().zip(fan_out.iter()) {
                if arm.members.is_empty() {
                    to_rewrite.push(SeedArc {
                        arc: out.arc,
                        target: out.target,
                    });
                    continue;
                }
                for &member in &arm.members {
                    for boundary in view.arcs_out(member) {
                        if !partition.arm_members.contains(&boundary.target) {
                            to_rewrite.push(SeedArc {
                                arc: boundary.arc,
                                target: boundary.target,
                            });
                        }
                    }
                }
            }
        }
        for out_arc in to_rewrite {
            self.overlay.rewrite_arc(
                out_arc.arc,
                &[AuxAssign {
                    var: selector,
                    value: selector_value(out_arc.target),
                }],
                Vertex::Aux(demux),
            );
        }
        demux
    }

    fn out_arcs(&self, vertex: Vertex, scope: &Scope<'_>) -> SmallVec<[SeedArc; 4]> {
        let view = make_view(self.mapper, self.overlay, &self.set_pool, scope);
        view.arcs_out(vertex)
            .iter()
            .map(|traversed| SeedArc {
                arc: traversed.arc,
                target: traversed.target,
            })
            .collect()
    }

    fn is_member(&self, vertex: Vertex, scope: &Scope<'_>) -> bool {
        match scope.members {
            None => true,
            Some(index) => self.set_pool[index].contains(vertex),
        }
    }

    /// A vertex inserted for the current region (a demux or a promoted
    /// assignment) joins that region's member set. The universal root has
    /// no set to maintain.
    fn add_region_member(&mut self, vertex: Vertex, scope: &Scope<'_>) {
        if let Some(index) = scope.members {
            self.set_pool[index].insert(vertex);
        }
    }

    fn ensure_pool(&mut self, depth: usize) {
        while self.set_pool.len() <= depth {
            self.set_pool
                .push(VertexSet::new(self.mapper.blocks.len(), self.tree.len()));
        }
    }

    fn fill_pool_set(&mut self, depth: usize, members: &[Vertex]) {
        self.ensure_pool(depth);
        let set = &mut self.set_pool[depth];
        set.clear();
        for &member in members {
            set.insert(member);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::control_flow::build_overlay;
    use crate::llvm_parser::scc::SccTree;
    use llvm_ir::Module;
    use std::sync::Mutex;

    // llvm-ir lazily initialises a global attribute table on first parse
    // that races under concurrent test threads; serialise parses.
    static LLVM_PARSE_LOCK: Mutex<()> = Mutex::new(());

    struct Prepared {
        mapper: BasicBlockMapper,
        overlay: Overlay,
    }

    /// Parse `ir` and run both restructuring passes on the first function.
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
            .chain(std::iter::once(false))
            .collect();
        let tree = SccTree::build(&mapper);
        let overlay = build_overlay(&mapper, &tree, diverging);
        Prepared { mapper, overlay }
    }

    fn block_named(prepared: &Prepared, name: &str) -> BasicBlockId {
        *prepared
            .mapper
            .get(&llvm_ir::Name::Name(Box::new(name.to_string())))
            .unwrap_or_else(|| panic!("no block named {name}"))
    }

    fn demux_ids(prepared: &Prepared) -> Vec<AuxVertexId> {
        prepared
            .overlay
            .aux_vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(v.kind, AuxVertexKind::BranchDemux))
            .map(|(i, _)| AuxVertexId(i as u32))
            .collect()
    }

    fn promoted_ids(prepared: &Prepared) -> Vec<AuxVertexId> {
        prepared
            .overlay
            .aux_vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(v.kind, AuxVertexKind::PromotedAssign { .. }))
            .map(|(i, _)| AuxVertexId(i as u32))
            .collect()
    }

    #[test]
    fn diamond_with_single_join_inserts_nothing() {
        let ir = r#"
define i32 @f(i1 %c) {
entry:
  br i1 %c, label %t, label %e
t:
  br label %m
e:
  br label %m
m:
  %r = phi i32 [ 10, %t ], [ 20, %e ]
  ret i32 %r
}
"#;
        let prepared = run_on(ir);
        assert!(prepared.overlay.aux_vertices.is_empty());
        assert!(prepared.overlay.rewrites.is_empty());
    }

    #[test]
    fn multi_continuation_branch_gets_one_demux_and_staged_joins_none() {
        // The branch at entry has two continuation points (contB and
        // contC): a demux routes them. The demux's own partition then has a
        // single continuation, so no second demux appears.
        let ir = r#"
define i32 @f(i32 %a, i32 %b, i32 %x) {
entry:
  %c0 = icmp ne i32 %a, 0
  br i1 %c0, label %sw, label %contB
sw:
  switch i32 %b, label %toC [ i32 7, label %toB ]
toB:
  br label %contB
toC:
  br label %contC
contB:
  %vb = mul i32 %x, 3
  br label %contC
contC:
  %r = phi i32 [ %vb, %contB ], [ %x, %toC ]
  ret i32 %r
}
"#;
        let prepared = run_on(ir);
        let demuxes = demux_ids(&prepared);
        assert_eq!(1, demuxes.len(), "staged joins need exactly one demux");
        let demux = demuxes[0];

        let cont_b = block_named(&prepared, "contB");
        let cont_c = block_named(&prepared, "contC");
        let fan_out = &prepared.overlay.aux_vertices[demux.0 as usize].fan_out;
        assert_eq!(Vertex::Block(cont_b), fan_out[0].target);
        assert_eq!(Vertex::Block(cont_c), fan_out[1].target);

        // toB -> contB carries selector := 0; toC -> contC selector := 1;
        // the empty alternative (entry's false arc straight to contB)
        // carries selector := 0 on the fan-out arc itself.
        let selector = AuxVar::ContinuationSelector(demux);
        let check = |source: BasicBlockId, index: u32, value: u32| {
            let rewrite = prepared
                .overlay
                .rewrite_of(ArcId {
                    source: Vertex::Block(source),
                    index,
                })
                .unwrap_or_else(|| panic!("arc {source:?}/{index} rewritten"));
            assert_eq!(
                &[AuxAssign {
                    var: selector,
                    value,
                }][..],
                &rewrite.assignments[..]
            );
            assert_eq!(Vertex::Aux(demux), rewrite.redirect);
        };
        check(block_named(&prepared, "toB"), 0, 0);
        check(block_named(&prepared, "toC"), 0, 1);
        check(block_named(&prepared, "entry"), 1, 0);
        // The in-region join's own arc (contB -> contC) is tail-side and
        // stays untouched.
        assert!(
            prepared
                .overlay
                .rewrite_of(ArcId {
                    source: Vertex::Block(cont_b),
                    index: 0,
                })
                .is_none()
        );
    }

    #[test]
    fn trimming_promotes_an_arm_exit_cluster_before_the_demux() {
        // Inside the loop body, the branch at `body` has two continuation
        // points: the in-body join `j` and the loop tail (via x's exit arc,
        // which carries the loop's selector and repeat assignments). The
        // tail also has non-arm predecessors (the latch's repetition arc and
        // the header's exit arc), so trimming must lift x's cluster into a
        // PromotedAssign vertex and the demux funnel lands before it.
        let ir = r#"
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
"#;
        let prepared = run_on(ir);
        let scc = SccTreeNodeId(0);
        let record = prepared.overlay.loops[0].as_ref().expect("loop record");
        let tail = record.tail.expect("multi-exit loop is restructured");

        // Exit targets sorted ascending: out (declared first) then out2.
        let out = block_named(&prepared, "out");
        let out2 = block_named(&prepared, "out2");
        assert_eq!(
            &[Vertex::Block(out), Vertex::Block(out2)][..],
            &record.exit_targets[..]
        );

        // The promoted vertex holds x's whole exit cluster (selector := 1
        // for out2, repeat := 0) and leads to the tail.
        let promoted = {
            let ids = promoted_ids(&prepared);
            assert_eq!(1, ids.len(), "exactly one promotion");
            ids[0]
        };
        let promoted_vertex = &prepared.overlay.aux_vertices[promoted.0 as usize];
        let AuxVertexKind::PromotedAssign { assignments } = &promoted_vertex.kind else {
            unreachable!();
        };
        assert_eq!(
            &[
                AuxAssign {
                    var: AuxVar::LoopVertexSelector(scc),
                    value: 1,
                },
                AuxAssign {
                    var: AuxVar::LoopRepeat(scc),
                    value: 0,
                },
            ][..],
            &assignments[..]
        );
        assert_eq!(1, promoted_vertex.fan_out.len());
        assert_eq!(Vertex::Aux(tail), promoted_vertex.fan_out[0].target);

        // x's exit arc now carries ONLY the continuation selector and
        // funnels through the demux; the demux routes alternative 1 to the
        // promoted vertex (continuations sorted: Block(j) before Aux).
        let demuxes = demux_ids(&prepared);
        assert_eq!(1, demuxes.len());
        let demux = demuxes[0];
        let j = block_named(&prepared, "j");
        let fan_out = &prepared.overlay.aux_vertices[demux.0 as usize].fan_out;
        assert_eq!(Vertex::Block(j), fan_out[0].target);
        assert_eq!(Vertex::Aux(promoted), fan_out[1].target);

        let x = block_named(&prepared, "x");
        let x_exit = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(x),
                index: 0,
            })
            .expect("x's exit arc rewritten");
        assert_eq!(
            &[AuxAssign {
                var: AuxVar::ContinuationSelector(demux),
                value: 1,
            }][..],
            &x_exit.assignments[..]
        );
        assert_eq!(Vertex::Aux(demux), x_exit.redirect);

        // The non-arm assignment carriers stay fused, not promoted: the
        // latch's repetition arc and the header's exit arc keep their
        // clusters.
        let latch = block_named(&prepared, "latch");
        let latch_rewrite = prepared
            .overlay
            .rewrite_of(ArcId {
                source: Vertex::Block(latch),
                index: 0,
            })
            .expect("repetition arc rewritten");
        assert_eq!(
            &[AuxAssign {
                var: AuxVar::LoopRepeat(scc),
                value: 1,
            }][..],
            &latch_rewrite.assignments[..]
        );
    }

    #[test]
    fn nonreconverging_returns_join_at_the_exit_block() {
        // Both alternatives return: the exit block is the single
        // continuation point and no demux is needed.
        let ir = r#"
define i32 @f(i1 %c) {
entry:
  br i1 %c, label %t, label %e
t:
  ret i32 11
e:
  ret i32 22
}
"#;
        let prepared = run_on(ir);
        assert!(prepared.overlay.aux_vertices.is_empty());
        assert!(prepared.overlay.rewrites.is_empty());
    }

    #[test]
    fn full_fixture_shapes_restructure_without_panicking() {
        // A switch inside a loop with mixed continuations (the shape of
        // tests/fixtures/c/28 and 38): exercises loop pass + branch pass
        // together, including a demux inside a loop body.
        let ir = r#"
define i32 @f(i32 %n) {
entry:
  br label %h
h:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %c = icmp slt i32 %i, %n
  br i1 %c, label %body, label %done
body:
  %m = urem i32 %i, 3
  switch i32 %m, label %a [ i32 1, label %b
                            i32 2, label %brk ]
a:
  br label %latch
b:
  br label %latch
brk:
  br label %after
latch:
  %i.next = add i32 %i, 1
  br label %h
done:
  br label %after
after:
  %r = phi i32 [ %i, %brk ], [ 0, %done ]
  ret i32 %r
}
"#;
        let prepared = run_on(ir);
        // Structural smoke test: the two passes compose on a loop whose
        // body mixes an in-body join (latch) with a break path (brk), and
        // whose two exit paths reconverge after the loop. The loop record
        // exists with an exit demux; the per-branch single-continuation
        // property is asserted for real by the emitter (step 4).
        let record = prepared.overlay.loops[0].as_ref().expect("loop record");
        assert!(record.exit_demux.is_some(), "two exit targets");
    }
}
