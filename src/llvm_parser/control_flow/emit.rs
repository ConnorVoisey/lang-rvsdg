//! **The RVSDG emitter** (the paper's BUILDRVSDG*, Bahmann et al. 2015
//! section 4): a single mechanical walk over the fully restructured control
//! flow graph.
//!
//! The construction order is the paper's, process-first and
//! assemble-afterwards (section 4's construction algorithm: "Recursively
//! process each alternative path. Afterwards, generate a gamma-node"):
//! each gamma alternative and each
//! theta body is emitted into its own fresh region under its own symbol
//! frame; the frame OBSERVES the subregion's reads of outer values (its
//! captures, which become the construct's inputs) and its writes (which
//! become the construct's outputs); the gamma or theta node is assembled
//! afterwards from those observations. Nothing is scanned from LLVM syntax:
//! a scan would be a second source of truth about what the walk binds, and
//! any disagreement is a silent miscompile.
//!
//! The walk's semantics per traversed arc, in fixed order: apply the arc's
//! phi copies (PARALLEL copies: resolve every incoming operand against the
//! scope as it stood before the arc, then write all destinations -- a
//! block's phis may reference each other's destinations to mean the
//! previous iteration's values), then bind the arc's auxiliary assignments
//! as integer constants, then move to the target. A region never walks a
//! non-member vertex: it applies the leaving arc's payload and returns.
//!
//! By the time this runs, `build_overlay` has given every branch exactly
//! one continuation point (asserted here) and made every loop body acyclic,
//! so the walk needs no analysis beyond re-partitioning each branch to
//! recover its alternatives' member sets.

use std::cell::RefCell;

use llvm_ir::{Instruction, Name};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::BasicBlockId,
        control_flow::{
            analysis::signature::{phi_incoming_from, phi_instructions_at},
            overlay::{AuxAssign, AuxVar, AuxVertexKind, Overlay, Vertex},
            partition::{Partitioner, SeedArc},
            scopes::{Frame, RET_VAL, SymbolId},
            view::{Membership, PhiCopies, RegionView, TraversedArc},
        },
        instructions::RegionLowerer,
        scc::SccTreeNodeId,
    },
    rvsdg::{
        ConstValue, MatchArm, RVSDGMod, RegionId, State, ValueId, builder::RegionBuilder,
        func::FnResult, types::TypeRef,
    },
};

/// One region of the emission recursion: its member set (arm sets come from
/// partitioning, body sets from the loop records plus aux-vertex ownership;
/// the function root is universal), the collapse table of its nesting
/// level, and the enclosing loop whose repetition arcs are hidden.
struct EmitRegion<'a> {
    members: Option<&'a FxHashSet<Vertex>>,
    collapse: &'a [Option<SccTreeNodeId>],
    body_of: Option<SccTreeNodeId>,
}

impl EmitRegion<'_> {
    fn contains(&self, vertex: Vertex) -> bool {
        match self.members {
            None => true,
            Some(set) => set.contains(&vertex),
        }
    }
}

/// Emit one function's body from its restructuring overlay. The caller has
/// seeded the root frame with the function parameters.
#[tracing::instrument(name = "emit", skip_all, fields(blocks = lowerer.fn_ctx.bb_mapper.blocks.len()))]
pub(in crate::llvm_parser) fn emit_function_body(
    lowerer: &mut RegionLowerer<'_, '_, '_>,
    overlay: &Overlay,
    state: State,
) -> color_eyre::Result<FnResult> {
    let fn_ctx = lowerer.fn_ctx;
    let tree = fn_ctx.scc_tree;
    let emitter = Emitter {
        fn_ctx,
        overlay,
        partitioner: RefCell::new(Partitioner::new(fn_ctx.bb_mapper.blocks.len(), tree.len())),
        scratch: RefCell::new(EmitScratch::default()),
    };

    let root_collapse = tree.collapse_table(&tree.roots, fn_ctx.bb_mapper.blocks.len());
    let root = EmitRegion {
        members: None,
        collapse: &root_collapse,
        body_of: None,
    };
    let state = emitter.emit_region(lowerer, &root, Vertex::Block(BasicBlockId(0)), state)?;

    let values = if matches!(fn_ctx.func.return_type.as_ref(), llvm_ir::Type::VoidType) {
        Vec::new()
    } else {
        let value = match lowerer.scopes.resolve_id(lowerer.rb.graph, RET_VAL) {
            Some(value) => value,
            None => {
                // Every path diverges: the function result is unreachable.
                let ret_ty = lowerer
                    .rb
                    .graph
                    .types
                    .convert_type_ref(&fn_ctx.func.return_type, fn_ctx.llvm_mod)?;
                lowerer.rb.constant(ret_ty, ConstValue::Poison)
            }
        };
        vec![value]
    };
    Ok(FnResult { state, values })
}

struct Emitter<'m> {
    fn_ctx: &'m FnCtx<'m>,
    overlay: &'m Overlay,
    /// Interior-mutable so nested emission can re-partition while an outer
    /// partition's results are still alive; a partition call completes
    /// before any nested emission starts, so the borrow is never held
    /// across recursion.
    partitioner: RefCell<Partitioner>,
    /// Reused assembly buffers (see [`EmitScratch`]). Borrowed only during
    /// a gamma's or theta's assembly phase, which never nests: an inner
    /// construct's assembly completes during the outer construct's arm or
    /// body EMISSION, before the outer assembly begins.
    scratch: RefCell<EmitScratch>,
}

/// Assembly-phase buffers reused across every gamma and theta of a
/// function, so steady-state assembly allocates nothing: the symbol-union
/// containers keep their capacity across `clear()`.
#[derive(Default)]
struct EmitScratch {
    /// Dedupe set for the output/slot union.
    seen: FxHashSet<SymbolId>,
    /// Union of written symbols (gamma outputs / theta slots), in order.
    symbols: Vec<SymbolId>,
    /// Gamma input symbol -> index into `input_values`.
    input_index: FxHashMap<SymbolId, u32>,
    input_symbols: Vec<SymbolId>,
    input_values: Vec<ValueId>,
    output_types: Vec<TypeRef>,
    results: Vec<ValueId>,
}

impl<'m> Emitter<'m> {
    fn view<'v>(&'v self, region: &'v EmitRegion<'v>) -> RegionView<'v> {
        RegionView {
            mapper: self.fn_ctx.bb_mapper,
            overlay: self.overlay,
            members: match region.members {
                None => Membership::Universal,
                Some(set) => Membership::Set(set),
            },
            collapse: region.collapse,
            body_of: region.body_of,
        }
    }

    /// Walk one region from `entry`, emitting as it goes. Returns the state
    /// after the region; every value result is a symbol binding the
    /// caller's frame observes.
    fn emit_region(
        &self,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
        region: &EmitRegion<'_>,
        entry: Vertex,
        mut state: State,
    ) -> color_eyre::Result<State> {
        let exit_block = self.overlay.exit_block;
        let mut current = entry;
        loop {
            match current {
                Vertex::Block(block) => {
                    if block != exit_block {
                        state = self.lower_block_instructions(lowerer, state, block)?;
                    }
                }
                Vertex::Loop(scc) => {
                    state = self.emit_theta(lowerer, region, scc, state)?;
                }
                Vertex::Aux(aux_id) => {
                    if let AuxVertexKind::PromotedAssign { assignments } =
                        &self.overlay.aux_vertices[aux_id.0 as usize].kind
                    {
                        for assign in assignments {
                            let value = lowerer.rb.const_i32(assign.value as i32);
                            lowerer.scopes.bind_aux(assign.var, value);
                        }
                    }
                    // Demux kinds carry no computation; their branch is
                    // handled below through their fan-out arcs.
                }
            }

            let (arc_count, single) = {
                let view = self.view(region);
                let arcs = view.arcs_out(current);
                let single =
                    (arcs.len() == 1).then(|| (OwnedPayload::of(&arcs[0]), arcs[0].target));
                (arcs.len(), single)
            };
            match arc_count {
                0 => return Ok(state),
                1 => {
                    let (payload, target) = single.expect("collected above");
                    payload.apply(self, lowerer)?;
                    if !region.contains(target) {
                        return Ok(state);
                    }
                    current = target;
                }
                _ => {
                    let (next_state, join) = self.emit_gamma(lowerer, region, current, state)?;
                    state = next_state;
                    match join {
                        Some(join) if region.contains(join) => current = join,
                        // No continuation point (every alternative ends the
                        // region within itself) or an exterior one: the
                        // region is done either way.
                        _ => return Ok(state),
                    }
                }
            }
        }
    }

    /// Lower every non-phi instruction of `block`, threading state. Phis
    /// are bindings applied at arc traversal, never instructions.
    fn lower_block_instructions(
        &self,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
        mut state: State,
        block: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let bb = &self.fn_ctx.func.basic_blocks[block.0 as usize];
        for inst in &bb.instrs {
            if matches!(inst, Instruction::Phi(_)) {
                continue;
            }
            state = lowerer.lower_instruction(state, inst)?;
        }
        Ok(state)
    }

    /// The exit block's "phi": bind the return value from `from`'s `ret`
    /// operand. A diverging block binds nothing.
    fn bind_return_value(
        &self,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
        from: BasicBlockId,
    ) -> color_eyre::Result<()> {
        match &self.fn_ctx.func.basic_blocks[from.0 as usize].term {
            llvm_ir::Terminator::Ret(ret) => {
                if let Some(operand) = &ret.return_operand {
                    let value = lowerer.operand(operand)?;
                    lowerer.scopes.bind_id(RET_VAL, value);
                }
                Ok(())
            }
            llvm_ir::Terminator::Unreachable(_) => Ok(()),
            other => Err(color_eyre::eyre::eyre!(
                "arc into the exit block from a non-returning terminator: {other:?}"
            )),
        }
    }

    /// Emit the gamma for the branch at `branch`: process each alternative
    /// in its own region and frame, then assemble the node from the frames'
    /// captures (inputs) and writes (outputs). Returns the post-gamma state
    /// and the branch's single continuation point.
    fn emit_gamma(
        &self,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
        region: &EmitRegion<'_>,
        branch: Vertex,
        state: State,
    ) -> color_eyre::Result<(State, Option<Vertex>)> {
        let (partition, seed_payloads) = {
            let view = self.view(region);
            let arcs = view.arcs_out(branch);
            let seeds: SmallVec<[SeedArc; 4]> = arcs
                .iter()
                .map(|a| SeedArc {
                    arc: a.arc,
                    target: a.target,
                })
                .collect();
            let payloads: Vec<OwnedPayload> = arcs.iter().map(OwnedPayload::of).collect();
            let partition = self
                .partitioner
                .borrow_mut()
                .partition(&view, branch, &seeds);
            (partition, payloads)
        };
        // Zero continuations: every alternative ends the region within
        // itself. More than one would mean build_overlay failed to insert a
        // demux.
        if partition.continuations.len() > 1 {
            return Err(color_eyre::eyre::eyre!(
                "branch at {branch:?} has {} continuation points after restructuring; \
                 build_overlay should have inserted a demux",
                partition.continuations.len()
            ));
        }
        let join = partition.continuations.first().copied();

        let predicate = self.branch_predicate(lowerer, branch, partition.arms.len())?;

        // Per-arm member sets, prebuilt so the recursion only reads.
        let arm_sets: Vec<FxHashSet<Vertex>> = partition
            .arms
            .iter()
            .map(|arm| arm.members.iter().copied().collect())
            .collect();

        // Process each alternative in its own region under its own frame.
        let mut arm_regions: Vec<RegionId> = Vec::with_capacity(partition.arms.len());
        let mut arm_frames: Vec<Frame> = Vec::with_capacity(partition.arms.len());
        for (index, arm) in partition.arms.iter().enumerate() {
            let region_id = lowerer.rb.add_region(state);
            arm_regions.push(region_id);
            lowerer.scopes.push_frame(region_id);
            {
                let mut arm_rb = RegionBuilder::over(&mut *lowerer.rb.graph, region_id);
                let mut arm_lowerer =
                    RegionLowerer::new(&mut arm_rb, &mut *lowerer.scopes, self.fn_ctx);
                seed_payloads[index].apply(self, &mut arm_lowerer)?;
                if !arm.members.is_empty() {
                    let arm_region = EmitRegion {
                        members: Some(&arm_sets[index]),
                        collapse: region.collapse,
                        body_of: region.body_of,
                    };
                    self.emit_region(&mut arm_lowerer, &arm_region, arm.seed.target, state)?;
                }
            }
            arm_frames.push(lowerer.scopes.pop_frame());
        }

        // Assembly, afterwards. Outputs: the union of written symbols in
        // arm order (deterministic). Inputs: the union of captured symbols,
        // plus outputs already bound in the enclosing scope (so alternatives
        // that did not write one pass the enclosing value through). The
        // buffers are the emitter's reused scratch; assembly never nests,
        // so the borrow is never held across another construct's assembly.
        let mut scratch = self.scratch.borrow_mut();
        let EmitScratch {
            seen: output_seen,
            symbols: outputs,
            input_index,
            input_symbols,
            input_values,
            output_types,
            results,
        } = &mut *scratch;
        output_seen.clear();
        outputs.clear();
        input_index.clear();
        input_symbols.clear();
        input_values.clear();
        output_types.clear();

        for frame in &arm_frames {
            for &symbol in &frame.write_order {
                if output_seen.insert(symbol) {
                    outputs.push(symbol);
                }
            }
        }

        for frame in &arm_frames {
            for capture in &frame.captures {
                if !input_index.contains_key(&capture.symbol) {
                    input_index.insert(capture.symbol, input_symbols.len() as u32);
                    input_symbols.push(capture.symbol);
                    input_values.push(capture.outer);
                }
            }
        }
        for &symbol in outputs.iter() {
            if input_index.contains_key(&symbol) {
                continue;
            }
            if let Some(value) = lowerer.scopes.resolve_id(lowerer.rb.graph, symbol) {
                input_index.insert(symbol, input_symbols.len() as u32);
                input_symbols.push(symbol);
                input_values.push(value);
            }
        }

        // The poison type for an output an alternative never binds: taken
        // from a sibling's written value.
        for &symbol in outputs.iter() {
            let written = arm_frames.iter().find_map(|frame| {
                frame
                    .final_value(symbol)
                    .filter(|binding| binding.written)
                    .map(|binding| binding.value)
            });
            let value = written.expect("every output symbol comes from a write");
            output_types.push(lowerer.rb.graph.values[value.0 as usize].ty);
        }

        // Align every region's parameters to the canonical input order and
        // set its results.
        for (index, frame) in arm_frames.iter().enumerate() {
            let region_id = arm_regions[index];
            let graph: &mut RVSDGMod = lowerer.rb.graph;
            let mut params: Vec<ValueId> = Vec::with_capacity(input_symbols.len());
            for (i, &symbol) in input_symbols.iter().enumerate() {
                let existing = frame
                    .captures
                    .iter()
                    .find(|capture| capture.symbol == symbol)
                    .map(|capture| capture.param);
                let param = existing.unwrap_or_else(|| {
                    let ty = graph.values[input_values[i].0 as usize].ty;
                    graph.append_region_param(region_id, ty)
                });
                params.push(param);
            }
            graph.set_region_params(region_id, params.clone());

            results.clear();
            for (o, &symbol) in outputs.iter().enumerate() {
                let value = if let Some(binding) = frame.final_value(symbol) {
                    binding.value
                } else if let Some(&i) = input_index.get(&symbol) {
                    params[i as usize]
                } else {
                    RegionBuilder::over(graph, region_id)
                        .constant(output_types[o], ConstValue::Poison)
                };
                results.push(value);
            }
            graph.set_region_results(region_id, results);
        }

        let result = lowerer.rb.finish_gamma(
            predicate,
            state,
            input_values,
            &arm_regions,
            outputs.len() as u16,
        );
        for (o, &symbol) in outputs.iter().enumerate() {
            lowerer.scopes.bind_id(symbol, result.result(o as u16));
        }
        for frame in arm_frames {
            lowerer.scopes.recycle_frame(frame);
        }
        Ok((result.state, join))
    }

    /// The gamma predicate for `branch`: a block's terminator condition or
    /// a demux's selector symbol, converted through a match node emitted
    /// here, adjacent to its one consumer.
    fn branch_predicate(
        &self,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
        branch: Vertex,
        alternatives: usize,
    ) -> color_eyre::Result<ValueId> {
        match branch {
            Vertex::Block(block) => match &self.fn_ctx.func.basic_blocks[block.0 as usize].term {
                llvm_ir::Terminator::CondBr(cond_br) => {
                    let condition = lowerer.operand(&cond_br.condition)?;
                    Ok(lowerer.rb.bool_predicate(condition))
                }
                llvm_ir::Terminator::Switch(switch) => {
                    let (predicate, _targets) = lowerer.switch_predicate(switch)?;
                    Ok(predicate)
                }
                other => Err(color_eyre::eyre::eyre!(
                    "branching block with terminator {other:?}"
                )),
            },
            Vertex::Aux(aux_id) => {
                let var = match &self.overlay.aux_vertices[aux_id.0 as usize].kind {
                    AuxVertexKind::BranchDemux => AuxVar::ContinuationSelector(aux_id),
                    AuxVertexKind::LoopEntryDemux { scc }
                    | AuxVertexKind::LoopExitDemux { scc } => AuxVar::LoopVertexSelector(*scc),
                    other => {
                        return Err(color_eyre::eyre::eyre!(
                            "branching aux vertex of kind {other:?}"
                        ));
                    }
                };
                let selector = lowerer
                    .scopes
                    .resolve_aux(lowerer.rb.graph, var)
                    .ok_or_else(|| {
                        color_eyre::eyre::eyre!("demux selector {var:?} read before assignment")
                    })?;
                Ok(lowerer.rb.identity_match(selector, alternatives as u32))
            }
            Vertex::Loop(_) => unreachable!("a collapsed loop never branches"),
        }
    }

    /// Emit the theta for the collapsed loop `scc`: process the body in its
    /// own region and frame, then assemble the node. The loop variables are
    /// the frame's captures (pass-through unless also written) and writes
    /// (evolving values: entry phis bound on repetition arcs, boundary phis
    /// bound on exit arcs, values defined inside and read later, selectors
    /// crossing outward); nothing is scanned.
    fn emit_theta(
        &self,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
        region: &EmitRegion<'_>,
        scc: SccTreeNodeId,
        state: State,
    ) -> color_eyre::Result<State> {
        let _ = region;
        let tree = self.fn_ctx.scc_tree;
        let record = self.overlay.loops[scc.0 as usize]
            .as_ref()
            .expect("emission reached a loop the loop pass skipped");
        let entry = match record.entry_demux {
            Some(demux) => Vertex::Aux(demux),
            None => Vertex::Block(record.entries[0]),
        };
        let repeat_var = AuxVar::LoopRepeat(scc);
        let structured_back_edge = record.structured_back_edge;
        let is_restructured = record.tail.is_some();

        // The body's nesting level: direct children are collapsed, their
        // blocks are not body members; body members also include the aux
        // vertices restructuring placed inside this body. One shared
        // builder with the branch pass, so the two can never drift.
        let child_collapse = tree.collapse_table(
            &tree.children[scc.0 as usize],
            self.fn_ctx.bb_mapper.blocks.len(),
        );
        let mut body_members: FxHashSet<Vertex> = FxHashSet::default();
        self.overlay
            .for_each_body_member(tree, scc, &child_collapse, |vertex| {
                body_members.insert(vertex);
            });

        let body_region_id = lowerer.rb.add_region(state);
        lowerer.scopes.push_frame(body_region_id);
        let condition = {
            let mut body_rb = RegionBuilder::over(&mut *lowerer.rb.graph, body_region_id);
            let mut body_lowerer =
                RegionLowerer::new(&mut body_rb, &mut *lowerer.scopes, self.fn_ctx);
            let body_region = EmitRegion {
                members: Some(&body_members),
                collapse: &child_collapse,
                body_of: Some(scc),
            };
            self.emit_region(&mut body_lowerer, &body_region, entry, state)?;

            // A structured loop's hidden back edge still defines the
            // next-iteration values: apply its phi copies at the body's end.
            if structured_back_edge.is_some() {
                let payload = {
                    let view = self.view(&body_region);
                    view.hidden_back_edge().map(|arc| OwnedPayload::of(&arc))
                };
                if let Some(payload) = payload {
                    payload.apply(self, &mut body_lowerer)?;
                }
            }

            // The repetition predicate: alternative 1 repeats.
            if is_restructured {
                let repeat = body_lowerer
                    .scopes
                    .resolve_aux(body_lowerer.rb.graph, repeat_var)
                    .ok_or_else(|| {
                        color_eyre::eyre::eyre!("loop repeat flag unbound at body end")
                    })?;
                body_lowerer.rb.match_op(
                    repeat,
                    &[MatchArm {
                        value: 1,
                        alternative: 1,
                    }],
                    0,
                    2,
                )
            } else {
                // Already-structured loop: the tail block's branch
                // condition, oriented so taking the back edge means repeat.
                let back_edge = structured_back_edge
                    .expect("a loop is either restructured or has a recorded back edge");
                let Vertex::Block(tail_block) = back_edge.source else {
                    unreachable!("a structured back edge starts at a block");
                };
                let condition = match &self.fn_ctx.func.basic_blocks[tail_block.0 as usize].term {
                    llvm_ir::Terminator::CondBr(cond_br) => {
                        body_lowerer.operand(&cond_br.condition)?
                    }
                    other => {
                        return Err(color_eyre::eyre::eyre!(
                            "structured loop tail with terminator {other:?}"
                        ));
                    }
                };
                // CondBr arc 0 is the true alternative. If the back edge is
                // arc 0, condition true means repeat; otherwise false does.
                let repeat_when = i64::from(back_edge.index == 0);
                body_lowerer.rb.match_op(
                    condition,
                    &[MatchArm {
                        value: repeat_when,
                        alternative: 1,
                    }],
                    0,
                    2,
                )
            }
        };
        let frame = lowerer.scopes.pop_frame();

        // Assembly, afterwards. Slots: the frame's captures in capture
        // order, then written-only symbols in first-write order. Buffers
        // come from the emitter's reused scratch (assembly never nests).
        let mut scratch = self.scratch.borrow_mut();
        let EmitScratch {
            seen: slot_seen,
            symbols: slots,
            input_values: loop_var_inputs,
            results: next_values,
            ..
        } = &mut *scratch;
        slot_seen.clear();
        slots.clear();
        loop_var_inputs.clear();
        next_values.clear();

        for capture in &frame.captures {
            if slot_seen.insert(capture.symbol) {
                slots.push(capture.symbol);
            }
        }
        for &symbol in &frame.write_order {
            if slot_seen.insert(symbol) {
                slots.push(symbol);
            }
        }

        let mut params: Vec<ValueId> = Vec::with_capacity(slots.len());
        for &symbol in slots.iter() {
            let existing_capture = frame
                .captures
                .iter()
                .find(|capture| capture.symbol == symbol);
            let final_value = frame
                .final_value(symbol)
                .expect("every slot symbol is bound in the body frame")
                .value;
            next_values.push(final_value);
            match existing_capture {
                Some(capture) => {
                    loop_var_inputs.push(capture.outer);
                    params.push(capture.param);
                }
                None => {
                    // Written-only: the initial value is the enclosing
                    // binding (a value that already existed before the
                    // loop), or poison when there is none (for example
                    // another entry vertex's phis on the entering path).
                    let ty = lowerer.rb.graph.values[final_value.0 as usize].ty;
                    let init = match lowerer.scopes.resolve_id(lowerer.rb.graph, symbol) {
                        Some(value) => value,
                        None => lowerer.rb.constant(ty, ConstValue::Poison),
                    };
                    loop_var_inputs.push(init);
                    params.push(lowerer.rb.graph.append_region_param(body_region_id, ty));
                }
            }
        }
        lowerer.rb.graph.set_region_params(body_region_id, params);
        lowerer
            .rb
            .graph
            .set_region_results(body_region_id, next_values);

        let result = lowerer
            .rb
            .finish_theta(state, loop_var_inputs, body_region_id, condition);

        // Only written symbols rebind (a capture that was never written
        // just passed through; its enclosing binding is still the value).
        for (index, &symbol) in slots.iter().enumerate() {
            let written = frame
                .final_value(symbol)
                .is_some_and(|binding| binding.written);
            if written {
                lowerer.scopes.bind_id(symbol, result.result(index as u16));
            }
        }
        lowerer.scopes.recycle_frame(frame);
        Ok(result.state)
    }
}

/// An arc's payload, copied out of the view so it can be applied while the
/// graph is mutated.
struct OwnedPayload {
    phi_copies: Option<PhiCopies>,
    assignments: SmallVec<[AuxAssign; 3]>,
}

impl OwnedPayload {
    fn of(arc: &TraversedArc<'_>) -> Self {
        Self {
            phi_copies: arc.phi_copies,
            assignments: arc.assignments.iter().copied().collect(),
        }
    }

    /// Apply the payload: parallel phi copies (every incoming operand is
    /// resolved against the scope as it stood before this arc, then all
    /// destinations are written), then the auxiliary constant assignments.
    fn apply(
        &self,
        emitter: &Emitter<'_>,
        lowerer: &mut RegionLowerer<'_, '_, '_>,
    ) -> color_eyre::Result<()> {
        if let Some(copies) = self.phi_copies {
            if copies.block == emitter.overlay.exit_block {
                emitter.bind_return_value(lowerer, copies.from)?;
            } else {
                let bb = &emitter.fn_ctx.func.basic_blocks[copies.block.0 as usize];
                let phis = phi_instructions_at(bb);
                let mut resolved: SmallVec<[(&Name, ValueId); 4]> = SmallVec::new();
                for phi in &phis {
                    if let Some((operand, _)) =
                        phi_incoming_from(phi, emitter.fn_ctx.bb_mapper, |p| p == copies.from)
                    {
                        resolved.push((&phi.dest, lowerer.operand(operand)?));
                    }
                }
                for (name, value) in resolved {
                    lowerer.scopes.bind_name(name, value);
                }
            }
        }
        for assign in &self.assignments {
            let value = lowerer.rb.const_i32(assign.value as i32);
            lowerer.scopes.bind_aux(assign.var, value);
        }
        Ok(())
    }
}
