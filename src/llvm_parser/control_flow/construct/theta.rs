//! Construction of loop (theta) nodes: reducible and multi-entry loop setup, the
//! entry-region capture, the per-iteration body walker (and its in-body
//! route/demux), the leaf builders, and the post-theta exit demux.

use llvm_ir::{Name, Operand, TypeRef as LLVMTypeRef, instruction::Phi};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        block_mapper::BasicBlockId,
        control_flow::{
            analysis::signature::{
                collect_walked_blocks, phi_incoming_from, phi_instructions_at, region_live_ins,
            },
            construct::{ConstructExit, TargetCapture, branch_refs, seed_params},
            restructure::arm_target_blocks,
            rst::{
                DemuxBranchTarget, EntryExit, EntryRegion, ExitDemux, ExitMerge, LoopBodyExit,
                LoopBodyRegion, LoopCaptureExit, LoopCaptureRegion, SeqRegion, ThetaKind,
                ThetaNode,
            },
        },
        instructions::{
            RegionLowerer, for_each_operand, for_each_terminator_operand, instruction_dest,
        },
    },
    rvsdg::{
        MatchArm, State, ValueId,
        builder::{BranchResult, LoopResult, RegionBuilder, ThetaResult},
        types::{I32, TypeRef, VOID},
        value::ConstValue,
    },
};

/// The signature of a loop being lowered: the per-iteration leaf shape and the
/// data each leaf needs. The body walker is shared across both kinds; only the
/// leaf production and the theta setup differ.
struct LoopCtx<'m> {
    /// The loop-body region boundary: a back-edge to one of these stops as a
    /// repeat, an exit-arc target stops as an exit.
    boundary: SmallVec<[BasicBlockId; 8]>,
    /// The type of each leaf slot, in order (loop-variable slots, then `r`).
    /// Used to capture/forward a full loop leaf for a boundary demux target.
    leaf_types: Vec<TypeRef>,
    kind: LoopKind<'m>,
}

enum LoopKind<'m> {
    /// A single-entry loop.
    Reducible(ReducibleLoop<'m>),
    /// A multi-entry (irreducible) loop.
    MultiEntry(MultiEntryLoop<'m>),
}

/// A single-entry loop's leaf layout. Leaf slots: header phis, body live-ins,
/// loop-closed values, one per exit phi, optional exit `q`, then `r`.
struct ReducibleLoop<'m> {
    header: BasicBlockId,
    exit_blocks: Vec<BasicBlockId>,
    header_phis: Vec<&'m Phi>,
    header_dests: Vec<Name>,
    live_in_names: Vec<Name>,
    /// Values defined in the loop and used after it that are not header phis
    /// (loop-closed values; with the IR not in loop-closed SSA form they are
    /// used directly, so they are detected and carried out as extra slots).
    closed: Vec<Name>,
    exit_phis: Vec<(BasicBlockId, &'m Phi, TypeRef)>,
    has_exit_q: bool,
}

/// A multi-entry (irreducible) loop's leaf layout. Leaf slots: every entry
/// vertex's phis, loop-closed values, the entry `q` (which entry to resume), the
/// exit `q` (which exit was taken), then `r`. `base` = entry phis + closed values.
struct MultiEntryLoop<'m> {
    entries: Vec<BasicBlockId>,
    entry_phis: Vec<(BasicBlockId, &'m Phi)>,
    closed: Vec<Name>,
    exit_targets: Vec<BasicBlockId>,
    base: usize,
}

impl ReducibleLoop<'_> {
    /// Leaf length (loop-variable slots plus `r`).
    fn arity(&self) -> usize {
        self.header_phis.len()
            + self.live_in_names.len()
            + self.closed.len()
            + self.exit_phis.len()
            + self.has_exit_q as usize
            + 1
    }
}

impl MultiEntryLoop<'_> {
    /// Leaf length (loop-variable slots plus `r`).
    fn arity(&self) -> usize {
        self.base + 3
    }
}

impl LoopCtx<'_> {
    /// Leaf length (loop-variable slots plus `r`).
    fn arity(&self) -> usize {
        match &self.kind {
            LoopKind::Reducible(reducible) => reducible.arity(),
            LoopKind::MultiEntry(multi) => multi.arity(),
        }
    }

    fn boundary(&self) -> &[BasicBlockId] {
        &self.boundary
    }
}

/// The context threaded through a loop-body demux's head-arm walk
/// ([`RegionLowerer::construct_loop_capture`]): the demux continuation targets
/// with their capture layout, the total captured-slot count, and the enclosing
/// loop. Bundled so the recursive walk passes one reference instead of four.
struct LoopDemuxCtx<'a, 'm> {
    targets: &'a [DemuxBranchTarget],
    captures: &'a [TargetCapture<'m>],
    captured_count: usize,
    loop_ctx: &'a LoopCtx<'m>,
}

/// The result of building a single-entry loop's theta (without the post-theta
/// exit dispatch): the theta node and the slot layout the dispatch needs. Shared
/// by the reconverging path ([`RegionLowerer::construct_reducible_theta`]) and
/// the terminal path ([`RegionLowerer::construct_loop_return`]).
struct ReducibleThetaBuild {
    result: ThetaResult,
    /// Theta-result slot holding the exit `q` (valid only when the loop has more
    /// than one exit vertex; otherwise there is no exit dispatch).
    exit_q_slot: usize,
    /// Loop-closed exit-phi values bound after the theta, as `(name, value)`, to
    /// seed exit-dispatch arms that reference them.
    exit_phi_seeds: Vec<(Name, ValueId)>,
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Emit a theta node for `theta`, returning the post-theta state.
    pub(in crate::llvm_parser) fn construct_theta(
        &mut self,
        theta: &ThetaNode,
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        match &theta.kind {
            ThetaKind::Reducible { header, body } => {
                self.construct_reducible_theta(theta, *header, body, state, boundary)
            }
            ThetaKind::MultiEntry {
                entries,
                entry_region,
                bodies,
            } => self.construct_multi_entry_theta(
                theta,
                entries,
                entry_region,
                bodies,
                state,
                boundary,
            ),
        }
    }

    /// Emit a single-entry loop's theta, then its post-theta exit dispatch (the
    /// reconvergence demux, if any). Used for a loop that is a mid-region item;
    /// a loop that terminates its region goes through [`Self::construct_loop_return`].
    fn construct_reducible_theta(
        &mut self,
        theta: &ThetaNode,
        header: BasicBlockId,
        body: &LoopBodyRegion,
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        let build = self.build_reducible_theta(theta, header, body, state)?;
        match &theta.exit_demux {
            None => Ok(build.result.state),
            Some(exit_demux) => {
                let exit_q = build.result.result(build.exit_q_slot as u16);
                let exit_q_ctrl = self
                    .rb
                    .identity_match(exit_q, theta.exit_blocks.len() as u32);
                self.construct_exit_demux(
                    exit_q_ctrl,
                    build.result.state,
                    &theta.exit_blocks,
                    exit_demux,
                    &build.exit_phi_seeds,
                    boundary,
                )
            }
        }
    }

    /// Build a single-entry loop's theta: lay out the per-iteration leaf (header
    /// phis, body live-ins, loop-closed values, exit phis, optional exit `q`),
    /// emit the theta, and bind each loop-closed value to its theta output in this
    /// scope. Returns the theta result plus the slot layout the post-theta exit
    /// dispatch reads. Does not emit the dispatch itself (the caller does, since
    /// it differs between a reconverging and a terminal loop).
    fn build_reducible_theta(
        &mut self,
        theta: &ThetaNode,
        header: BasicBlockId,
        body: &LoopBodyRegion,
        state: State,
    ) -> color_eyre::Result<ReducibleThetaBuild> {
        let scc_body: FxHashSet<BasicBlockId> = self.fn_ctx.scc_tree.blocks[theta.scc.0 as usize]
            .iter()
            .copied()
            .collect();
        let in_scc = |block: BasicBlockId| scc_body.contains(&block);

        // Header phis: the loop-carried variables (init = a non-SCC predecessor's
        // incoming; each leaf supplies the next-iteration value per latch).
        let header_phis: Vec<&Phi> =
            phi_instructions_at(&self.fn_ctx.func.basic_blocks[header.0 as usize]).to_vec();
        let mut header_inits: Vec<ValueId> = Vec::with_capacity(header_phis.len());
        let mut header_dests: Vec<Name> = Vec::with_capacity(header_phis.len());
        for phi in &header_phis {
            let init_operand = phi
                .incoming_values
                .iter()
                .find(|(_, pred_name)| {
                    self.fn_ctx
                        .bb_mapper
                        .get(pred_name)
                        .is_some_and(|&id| !in_scc(id))
                })
                .map(|(operand, _)| operand)
                .ok_or_else(|| {
                    color_eyre::eyre::eyre!("header phi {:?} has no preheader incoming", phi.dest)
                })?;
            header_inits.push(self.operand(init_operand)?);
            header_dests.push(phi.dest.clone());
        }

        let scc_body_vec: Vec<BasicBlockId> = scc_body.iter().copied().collect();
        let (live_in_names, live_in_values) =
            region_live_ins(self.fn_ctx, &self.name_to_value, &scc_body_vec, &[], None);

        // Phis at the demux reconvergence are join phis (resolved by the exit
        // demux), not loop-closed exit phis -- exclude them here so they are not
        // double-handled when an exit arc targets the join directly. Only a
        // reconverging demux has such a join; a terminal one carries no join.
        let demux_join = match theta.exit_demux.as_ref().map(|demux| &demux.merge) {
            Some(ExitMerge::Reconverge { join }) => Some(*join),
            _ => None,
        };
        let mut exit_phis: Vec<(BasicBlockId, &Phi, TypeRef)> = Vec::new();
        for &exit_block in &theta.exit_blocks {
            if Some(exit_block) == demux_join {
                continue;
            }
            for phi in phi_instructions_at(&self.fn_ctx.func.basic_blocks[exit_block.0 as usize]) {
                let ty = self
                    .rb
                    .graph
                    .types
                    .convert_type_ref(&phi.to_type, self.fn_ctx.llvm_mod)?;
                exit_phis.push((exit_block, phi, ty));
            }
        }
        let has_exit_q = theta.exit_blocks.len() > 1;

        // Loop-closed values: defined in the loop body and used after it. Header
        // phis are excluded (they are carried as their own slots and bound below);
        // the rest are carried out as extra slots, since with the IR not in
        // loop-closed SSA form they are used directly (no exit phi to carry them).
        let header_dest_set: FxHashSet<&Name> = header_dests.iter().collect();
        let closed_typed: Vec<(Name, LLVMTypeRef)> = self
            .collect_loop_closed(&scc_body)
            .into_iter()
            .filter(|(name, _)| !header_dest_set.contains(name))
            .collect();
        let mut closed: Vec<Name> = Vec::with_capacity(closed_typed.len());
        let mut closed_inits: Vec<ValueId> = Vec::with_capacity(closed_typed.len());
        for (name, llvm_ty) in &closed_typed {
            let ty = self
                .rb
                .graph
                .types
                .convert_type_ref(llvm_ty, self.fn_ctx.llvm_mod)?;
            closed_inits.push(self.rb.constant(ty, ConstValue::Poison));
            closed.push(name.clone());
        }

        let n_header = header_inits.len();
        let n_live = live_in_values.len();
        let n_closed = closed.len();
        let n_exit_phi = exit_phis.len();

        let mut loop_var_inits: Vec<ValueId> =
            Vec::with_capacity(n_header + n_live + n_closed + n_exit_phi + has_exit_q as usize);
        loop_var_inits.extend_from_slice(&header_inits);
        loop_var_inits.extend_from_slice(&live_in_values);
        loop_var_inits.extend_from_slice(&closed_inits);
        for (_, _, ty) in &exit_phis {
            loop_var_inits.push(self.rb.constant(*ty, ConstValue::Poison));
        }
        if has_exit_q {
            loop_var_inits.push(self.rb.constant(I32, ConstValue::Poison));
        }

        let leaf_types: Vec<TypeRef> = loop_var_inits
            .iter()
            .map(|&v| self.rb.graph.values[v.0 as usize].ty)
            .chain(std::iter::once(I32))
            .collect();

        let mut loop_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
        loop_boundary.push(header);
        loop_boundary.extend_from_slice(&theta.exit_blocks);

        // The body closure seeds these names and the post-theta binding reads
        // them; cloned out before the layout is moved into `ctx` so neither has
        // to round-trip through the `LoopKind` enum.
        let seed_header = header_dests.clone();
        let seed_live = live_in_names.clone();
        let seed_closed = closed.clone();
        let bind_exit_phis: Vec<(BasicBlockId, &Phi, TypeRef)> = exit_phis.clone();

        let ctx = LoopCtx {
            boundary: loop_boundary,
            leaf_types,
            kind: LoopKind::Reducible(ReducibleLoop {
                header,
                exit_blocks: theta.exit_blocks.clone(),
                header_phis,
                header_dests,
                live_in_names,
                closed,
                exit_phis,
                has_exit_q,
            }),
        };

        let fn_ctx = self.fn_ctx;
        let ctx_ref = &ctx;
        let result = self.rb.theta(state, &loop_var_inits, |body_rb| {
            let mut ntv = FxHashMap::default();
            seed_params(body_rb, &seed_header, 0, &mut ntv);
            seed_params(body_rb, &seed_live, n_header as u32, &mut ntv);
            seed_params(body_rb, &seed_closed, (n_header + n_live) as u32, &mut ntv);
            let mut body_lowerer = RegionLowerer::new_child(body_rb, fn_ctx, ntv);
            let (next_state, leaf) =
                body_lowerer.construct_loop_body(body, state, None, ctx_ref)?;
            // The last leaf slot is the repetition predicate: 1 = iterate again,
            // 0 = leave the loop. Everything before it is the next-iteration vars.
            let repeat_pred = leaf[leaf.len() - 1];
            let next_vars = leaf[..leaf.len() - 1].to_vec();
            let condition = body_lowerer.rb.match_op(
                repeat_pred,
                &[MatchArm {
                    value: 1,
                    alternative: 1,
                }],
                0,
                2,
            );
            Ok(LoopResult {
                condition,
                next_state,
                next_vars,
            })
        })?;

        // Bind each loop-closed value used after the loop to its theta output:
        // header phis (slots 0..n_header), the closed extras, and the exit phis.
        for (i, dest) in seed_header.iter().enumerate() {
            self.name_to_value
                .insert(dest.clone(), result.result(i as u16));
        }
        for (k, name) in seed_closed.iter().enumerate() {
            self.name_to_value
                .insert(name.clone(), result.result((n_header + n_live + k) as u16));
        }
        let mut exit_phi_seeds: Vec<(Name, ValueId)> = Vec::with_capacity(n_exit_phi);
        for (index, (_, phi, _)) in bind_exit_phis.iter().enumerate() {
            let slot = n_header + n_live + n_closed + index;
            let value = result.result(slot as u16);
            self.name_to_value.insert(phi.dest.clone(), value);
            exit_phi_seeds.push((phi.dest.clone(), value));
        }

        Ok(ReducibleThetaBuild {
            result,
            exit_q_slot: n_header + n_live + n_closed + n_exit_phi,
            exit_phi_seeds,
        })
    }

    /// Emit a multi-entry (irreducible) loop's theta. The entry region computes
    /// the initial entry `q` plus the entry-phi inits; inside the theta a gamma on
    /// the entry `q` dispatches to each entry vertex's body. Loop-closed values
    /// (defined inside, used outside) are carried as extra slots and bound after.
    fn construct_multi_entry_theta(
        &mut self,
        theta: &ThetaNode,
        entries: &[BasicBlockId],
        entry_region: &EntryRegion,
        bodies: &[LoopBodyRegion],
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        let mut entry_phis: Vec<(BasicBlockId, &Phi)> = Vec::new();
        for &entry in entries {
            for phi in phi_instructions_at(&self.fn_ctx.func.basic_blocks[entry.0 as usize]) {
                entry_phis.push((entry, phi));
            }
        }
        let n_entry_phi = entry_phis.len();

        let scc_body: FxHashSet<BasicBlockId> = self.fn_ctx.scc_tree.blocks[theta.scc.0 as usize]
            .iter()
            .copied()
            .collect();
        let closed_typed = self.collect_loop_closed(&scc_body);
        let mut closed: Vec<Name> = Vec::with_capacity(closed_typed.len());
        let mut closed_inits: Vec<ValueId> = Vec::with_capacity(closed_typed.len());
        for (name, llvm_ty) in &closed_typed {
            let ty = self
                .rb
                .graph
                .types
                .convert_type_ref(llvm_ty, self.fn_ctx.llvm_mod)?;
            closed_inits.push(self.rb.constant(ty, ConstValue::Poison));
            closed.push(name.clone());
        }
        let n_closed = closed.len();
        let base = n_entry_phi + n_closed;

        let (state, q_and_inits) =
            self.construct_entry_capture(entry_region, state, entries, &entry_phis)?;
        let q_init = q_and_inits[0];

        let mut inits: Vec<ValueId> = q_and_inits[1..].to_vec();
        inits.extend_from_slice(&closed_inits);
        inits.push(q_init);
        inits.push(self.rb.constant(I32, ConstValue::Int(0)));

        let leaf_types: Vec<TypeRef> = inits
            .iter()
            .map(|&v| self.rb.graph.values[v.0 as usize].ty)
            .chain(std::iter::once(I32))
            .collect();

        let mut body_boundary: SmallVec<[BasicBlockId; 8]> = entries.iter().copied().collect();
        for &exit_block in &theta.exit_blocks {
            if !body_boundary.contains(&exit_block) {
                body_boundary.push(exit_block);
            }
        }
        // Names each body arm seeds (entry-vertex phis then loop-closed values),
        // built before the layout is moved into `ctx`.
        let mut arm_seed_names: Vec<Name> = Vec::with_capacity(base);
        for (_, phi) in &entry_phis {
            arm_seed_names.push(phi.dest.clone());
        }
        for name in &closed {
            arm_seed_names.push(name.clone());
        }

        let ctx = LoopCtx {
            boundary: body_boundary,
            leaf_types,
            kind: LoopKind::MultiEntry(MultiEntryLoop {
                entries: entries.to_vec(),
                entry_phis,
                closed: closed.clone(),
                exit_targets: theta.exit_blocks.clone(),
                base,
            }),
        };

        let fn_ctx = self.fn_ctx;
        let ctx_ref = &ctx;
        let n_entries = entries.len();
        let seed_names_ref: &[Name] = &arm_seed_names;
        let result = self.rb.theta(state, &inits, |body_rb| {
            let q_entry = body_rb.param(base as u32);
            let live_in_vals: Vec<ValueId> = (0..base as u32).map(|i| body_rb.param(i)).collect();
            let arm_closures: Vec<_> = (0..n_entries)
                .map(|entry_index| {
                    move |arm_rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                        let mut arm = RegionLowerer::arm_child(arm_rb, fn_ctx, seed_names_ref);
                        let (arm_state, leaf) =
                            arm.construct_loop_body(&bodies[entry_index], state, None, ctx_ref)?;
                        Ok(BranchResult {
                            state: arm_state,
                            values: leaf,
                        })
                    }
                })
                .collect();
            let refs = branch_refs(&arm_closures);
            let q_entry_ctrl = body_rb.identity_match(q_entry, n_entries as u32);
            let gamma = body_rb.gamma_n(q_entry_ctrl, state, &live_in_vals, &refs)?;
            // Leaf layout: `base` loop-var slots, then entry `q`, exit `q`, and the
            // repetition predicate `r` -- so slot `base + 2` is `r`, and the
            // next-iteration vars are everything up to (not including) it.
            let leaf_len = base + 3;
            let values: Vec<ValueId> = (0..leaf_len as u16).map(|i| gamma.result(i)).collect();
            let next_vars = values[..base + 2].to_vec();
            let repeat_pred = values[base + 2];
            let condition = body_rb.match_op(
                repeat_pred,
                &[MatchArm {
                    value: 1,
                    alternative: 1,
                }],
                0,
                2,
            );
            Ok(LoopResult {
                condition,
                next_state: gamma.state,
                next_vars,
            })
        })?;

        for (index, name) in closed.iter().enumerate() {
            self.name_to_value
                .insert(name.clone(), result.result((n_entry_phi + index) as u16));
        }

        match &theta.exit_demux {
            None => Ok(result.state),
            Some(exit_demux) => {
                let q_exit = result.result((base + 1) as u16);
                let q_exit_ctrl = self
                    .rb
                    .identity_match(q_exit, theta.exit_blocks.len() as u32);
                self.construct_exit_demux(
                    q_exit_ctrl,
                    result.state,
                    &theta.exit_blocks,
                    exit_demux,
                    &[],
                    boundary,
                )
            }
        }
    }

    /// Walk the entry region of an irreducible loop, producing the leaf
    /// `[q, entry-phi inits..]`: reaching an entry vertex yields its `q` index and
    /// that entry's phi incomings (poison for the others); branches merge the
    /// tuples via gammas.
    fn construct_entry_capture(
        &mut self,
        region: &EntryRegion,
        state: State,
        entries: &[BasicBlockId],
        entry_phis: &[(BasicBlockId, &Phi)],
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let (state, _prev) = self.lower_items(&region.items, state, None, entries)?;
        match &region.exit {
            EntryExit::ToContinuation { reached, via } => {
                let leaf = self.entry_leaf(*reached, *via, entries, entry_phis)?;
                Ok((state, leaf))
            }
            EntryExit::Route { head, arms } => {
                let arm_targets = arm_target_blocks(self.fn_ctx, *head)?;
                let phis: Vec<&Phi> = entry_phis.iter().map(|(_, phi)| *phi).collect();
                let walked: Vec<BasicBlockId> = arm_targets
                    .iter()
                    .flat_map(|&target| collect_walked_blocks(self.fn_ctx, target, entries))
                    .collect();
                let (names, live_ins) = region_live_ins(
                    self.fn_ctx,
                    &self.name_to_value,
                    &walked,
                    &phis,
                    Some(*head),
                );
                let predicate = self.branch_predicate(*head)?;
                let width = 1 + entry_phis.len();

                let fn_ctx = self.fn_ctx;
                let names_ref: &[Name] = &names;
                let entries_ref = entries;
                let entry_phis_ref = entry_phis;

                let sub_closures: Vec<_> = arms
                    .iter()
                    .map(|arm_region| {
                        move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                            let mut sub = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                            let (arm_state, arm_values) = sub.construct_entry_capture(
                                arm_region,
                                state,
                                entries_ref,
                                entry_phis_ref,
                            )?;
                            Ok(BranchResult {
                                state: arm_state,
                                values: arm_values,
                            })
                        }
                    })
                    .collect();
                let refs = branch_refs(&sub_closures);
                let gamma = self.rb.gamma_n(predicate, state, &live_ins, &refs)?;
                let values: Vec<ValueId> = (0..width as u16).map(|i| gamma.result(i)).collect();
                Ok((gamma.state, values))
            }
        }
    }

    /// The entry leaf: control enters at `reached` from `via`. Yields that entry's
    /// `q` index, then per entry phi its incoming along this arc (for phis at
    /// `reached`) or poison (other entries' phis, set by a repetition before use).
    fn entry_leaf(
        &mut self,
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
        entries: &[BasicBlockId],
        entry_phis: &[(BasicBlockId, &Phi)],
    ) -> color_eyre::Result<Vec<ValueId>> {
        let src = via.ok_or_else(|| {
            color_eyre::eyre::eyre!("entry leaf reaching {} with no predecessor", reached.0)
        })?;
        let entry_index = entries
            .iter()
            .position(|&entry| entry == reached)
            .ok_or_else(|| color_eyre::eyre::eyre!("entry vertex {} not found", reached.0))?;
        let mut values: Vec<ValueId> = Vec::with_capacity(1 + entry_phis.len());
        values.push(self.rb.constant(I32, ConstValue::Int(entry_index as i64)));
        for (entry, phi) in entry_phis {
            if *entry == reached {
                let (operand, _) =
                    phi_incoming_from(phi, self.fn_ctx.bb_mapper, |block| block == src)
                        .ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "entry phi {:?} has no incoming for entry arc from {}",
                                phi.dest,
                                src.0
                            )
                        })?;
                values.push(self.operand(operand)?);
            } else {
                let ty = self
                    .rb
                    .graph
                    .types
                    .convert_type_ref(&phi.to_type, self.fn_ctx.llvm_mod)?;
                values.push(self.rb.constant(ty, ConstValue::Poison));
            }
        }
        Ok(values)
    }

    /// Loop-closed values of an irreducible loop: SSA values defined inside the
    /// SCC and used outside it (detected directly, since the IR is not in
    /// loop-closed SSA form). Returns each value's name and LLVM type.
    fn collect_loop_closed(&self, scc_body: &FxHashSet<BasicBlockId>) -> Vec<(Name, LLVMTypeRef)> {
        let mut defined_inside: FxHashSet<Name> = FxHashSet::default();
        for &block in scc_body {
            for inst in &self.fn_ctx.func.basic_blocks[block.0 as usize].instrs {
                if let Some(dest) = instruction_dest(inst) {
                    defined_inside.insert(dest.clone());
                }
            }
        }
        let mut closed: Vec<(Name, LLVMTypeRef)> = Vec::new();
        let mut seen: FxHashSet<Name> = FxHashSet::default();
        for (index, bb) in self.fn_ctx.func.basic_blocks.iter().enumerate() {
            if scc_body.contains(&BasicBlockId(index as u32)) {
                continue;
            }
            let mut visit = |operand: &Operand| {
                let Operand::LocalOperand { name, ty } = operand else {
                    return;
                };
                if defined_inside.contains(name) && seen.insert(name.clone()) {
                    closed.push((name.clone(), ty.clone()));
                }
            };
            for inst in &bb.instrs {
                for_each_operand(inst, &mut visit);
            }
            for_each_terminator_operand(&bb.term, &mut visit);
        }
        closed
    }

    /// Emit the post-theta exit demux of a multi-exit loop: a gamma on the exit
    /// `q` control that lowers each exit vertex's tail to the reconvergence
    /// `join`, binds `join`'s phis, and resumes there.
    fn construct_exit_demux(
        &mut self,
        exit_q_ctrl: ValueId,
        state: State,
        exit_blocks: &[BasicBlockId],
        exit_demux: &ExitDemux,
        exit_phi_seeds: &[(Name, ValueId)],
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        let join = match &exit_demux.merge {
            ExitMerge::Reconverge { join } => *join,
            ExitMerge::Return => {
                return Err(color_eyre::eyre::eyre!(
                    "construct_exit_demux called on a terminal (return) exit demux"
                ));
            }
        };
        let join_phis = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);

        let mut arm_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
        arm_boundary.push(join);
        arm_boundary.extend_from_slice(boundary);

        let (mut live_in_names, mut live_ins) =
            self.live_ins_for_arms(exit_blocks, &arm_boundary, &join_phis, None);
        // Each demux arm resolves the join phis along its own exit edge, whose
        // source may be a loop block (not in the walked tails) -- e.g. an exit arc
        // targeting the join directly. Add every join-phi incoming bound after the
        // theta (header phis, loop-closed values) as a live-in so the arm can
        // resolve it.
        let mut seen: FxHashSet<Name> = live_in_names.iter().cloned().collect();
        for phi in &join_phis {
            for (operand, _) in &phi.incoming_values {
                let Operand::LocalOperand { name, .. } = operand else {
                    continue;
                };
                if !seen.insert(name.clone()) {
                    continue;
                }
                if let Some(&value) = self.name_to_value.get(name) {
                    live_in_names.push(name.clone());
                    live_ins.push(value);
                }
            }
        }
        let phi_types = self.convert_phi_types(&join_phis)?;

        let mut inputs: Vec<ValueId> = live_ins;
        let n_live = live_in_names.len();
        for (_, value) in exit_phi_seeds {
            inputs.push(*value);
        }
        let seed_names: Vec<Name> = exit_phi_seeds
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        let fn_ctx = self.fn_ctx;
        let join_phis_ref: &[&Phi] = &join_phis;
        let phi_types_ref: &[TypeRef] = &phi_types;
        let live_names_ref: &[Name] = &live_in_names;
        let seed_names_ref: &[Name] = &seed_names;
        let arm_boundary_ref: &[BasicBlockId] = &arm_boundary;

        let arm_closures: Vec<_> = exit_demux
            .tails
            .iter()
            .map(|tail| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut ntv = FxHashMap::default();
                    seed_params(rb, live_names_ref, 0, &mut ntv);
                    seed_params(rb, seed_names_ref, n_live as u32, &mut ntv);
                    let mut arm = RegionLowerer::new_child(rb, fn_ctx, ntv);
                    let exit = arm.construct(tail, state, None, arm_boundary_ref)?;
                    arm.join_arm_result(exit, join, join_phis_ref, phi_types_ref, "exit-demux")
                }
            })
            .collect();
        let refs = branch_refs(&arm_closures);
        let result = self.rb.gamma_n(exit_q_ctrl, state, &inputs, &refs)?;
        for (i, phi) in join_phis.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), result.result(i as u16));
        }
        Ok(result.state)
    }

    /// Emit a multi-exit loop whose exit tails do not reconverge: build the theta,
    /// then dispatch the exit `q` through a return gamma over the tails (every one
    /// returning or diverging). Returns the post-loop state and the merged return
    /// values, which become the enclosing region's return.
    pub(in crate::llvm_parser) fn construct_loop_return(
        &mut self,
        theta: &ThetaNode,
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let (header, body) = match &theta.kind {
            ThetaKind::Reducible { header, body } => (*header, body),
            ThetaKind::MultiEntry { .. } => {
                return Err(color_eyre::eyre::eyre!(
                    "terminal exit dispatch for a multi-entry loop is not yet handled"
                ));
            }
        };
        let exit_demux = theta
            .exit_demux
            .as_ref()
            .ok_or_else(|| color_eyre::eyre::eyre!("loop-return theta has no exit demux"))?;

        let build = self.build_reducible_theta(theta, header, body, state)?;
        let exit_q = build.result.result(build.exit_q_slot as u16);
        let exit_q_ctrl = self
            .rb
            .identity_match(exit_q, theta.exit_blocks.len() as u32);
        self.construct_exit_return_gamma(
            exit_q_ctrl,
            build.result.state,
            &theta.exit_blocks,
            &exit_demux.tails,
            &build.exit_phi_seeds,
            boundary,
        )
    }

    /// Emit the post-theta return gamma of a non-reconverging multi-exit loop: a
    /// gamma on the exit `q` whose arm `i` lowers `tails[i]` (each returning or
    /// diverging) and produces the function return value(s). `exit_phi_seeds` are
    /// loop-closed values bound after the theta that a tail may reference.
    fn construct_exit_return_gamma(
        &mut self,
        exit_q_ctrl: ValueId,
        state: State,
        exit_blocks: &[BasicBlockId],
        tails: &[SeqRegion],
        exit_phi_seeds: &[(Name, ValueId)],
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let ret_ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&self.fn_ctx.func.return_type, self.fn_ctx.llvm_mod)?;
        let arity: u16 = if ret_ty == VOID { 0 } else { 1 };

        // Live-ins over the exit tails (bounded by the enclosing boundary), plus
        // the loop-closed exit-phi values bound after the theta that a tail uses.
        let (live_in_names, live_ins) = self.live_ins_for_arms(exit_blocks, boundary, &[], None);
        let mut inputs: Vec<ValueId> = live_ins;
        let n_live = live_in_names.len();
        for (_, value) in exit_phi_seeds {
            inputs.push(*value);
        }
        let seed_names: Vec<Name> = exit_phi_seeds
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        let fn_ctx = self.fn_ctx;
        let live_names_ref: &[Name] = &live_in_names;
        let seed_names_ref: &[Name] = &seed_names;
        let boundary_ref: &[BasicBlockId] = boundary;

        let arm_closures: Vec<_> = tails
            .iter()
            .map(|tail| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut ntv = FxHashMap::default();
                    seed_params(rb, live_names_ref, 0, &mut ntv);
                    seed_params(rb, seed_names_ref, n_live as u32, &mut ntv);
                    let mut arm = RegionLowerer::new_child(rb, fn_ctx, ntv);
                    match arm.construct(tail, state, None, boundary_ref)? {
                        ConstructExit::Returned { state, values } => {
                            Ok(BranchResult { state, values })
                        }
                        ConstructExit::Diverge { state } => {
                            let values = if arity == 0 {
                                Vec::new()
                            } else {
                                vec![arm.rb.constant(ret_ty, ConstValue::Poison)]
                            };
                            Ok(BranchResult { state, values })
                        }
                        ConstructExit::AtBoundary { reached, .. } => Err(color_eyre::eyre::eyre!(
                            "exit-return tail unexpectedly reached {}",
                            reached.0
                        )),
                    }
                }
            })
            .collect();
        let refs = branch_refs(&arm_closures);
        let result = self.rb.gamma_n(exit_q_ctrl, state, &inputs, &refs)?;
        let values: Vec<ValueId> = (0..arity).map(|i| result.result(i)).collect();
        Ok((result.state, values))
    }

    /// Walk a loop-body region, producing its per-iteration leaf vector. In-body
    /// reconvergences and straight-line blocks lower via `lower_items`; the
    /// region's exit produces the leaf (reaching the header is a repeat, an exit
    /// vertex is an exit, a router over loop boundaries is a leaf-merge gamma).
    fn construct_loop_body(
        &mut self,
        region: &LoopBodyRegion,
        state: State,
        entry_prev: Option<BasicBlockId>,
        ctx: &LoopCtx,
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let boundary: SmallVec<[BasicBlockId; 8]> = ctx.boundary().iter().copied().collect();
        let (state, _prev) = self.lower_items(&region.items, state, entry_prev, &boundary)?;
        match &region.exit {
            LoopBodyExit::ToContinuation { reached, via } => {
                let leaf = self.loop_leaf(*reached, *via, ctx)?;
                Ok((state, leaf))
            }
            LoopBodyExit::Route { head, arms } => {
                self.construct_loop_route(*head, arms, state, ctx)
            }
            LoopBodyExit::Demux {
                head,
                arms,
                targets,
            } => self.construct_loop_demux(*head, arms, targets, state, ctx),
        }
    }

    /// A loop-body router (a branch every continuation of which is a loop
    /// boundary): a gamma merging each arm's leaf into one per-iteration leaf.
    fn construct_loop_route(
        &mut self,
        head: BasicBlockId,
        arms: &[LoopBodyRegion],
        state: State,
        ctx: &LoopCtx,
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let arity = ctx.arity();
        let arm_targets = arm_target_blocks(self.fn_ctx, head)?;
        let walked: Vec<BasicBlockId> = arm_targets
            .iter()
            .flat_map(|&target| collect_walked_blocks(self.fn_ctx, target, ctx.boundary()))
            .collect();
        let (live_in_names, live_ins) = self.loop_live_ins_over(&walked, head, ctx);
        let predicate = self.branch_predicate(head)?;

        let fn_ctx = self.fn_ctx;
        let names_ref: &[Name] = &live_in_names;
        let ctx_ref = ctx;
        let route_head = head;

        let arm_closures: Vec<_> = arms
            .iter()
            .map(|arm_region| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut arm = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                    let (arm_state, leaf) =
                        arm.construct_loop_body(arm_region, state, Some(route_head), ctx_ref)?;
                    Ok(BranchResult {
                        state: arm_state,
                        values: leaf,
                    })
                }
            })
            .collect();
        let refs = branch_refs(&arm_closures);
        let result = self.rb.gamma_n(predicate, state, &live_ins, &refs)?;
        let leaf: Vec<ValueId> = (0..arity).map(|i| result.result(i as u16)).collect();
        Ok((result.state, leaf))
    }

    /// A loop-body `p`-demux whose continuations are a mix of in-body merges and
    /// loop boundaries: a head gamma discovers which continuation each arm reaches
    /// (`p`) and captures its data (an in-body continuation's phis, or a
    /// boundary's full leaf), and a demux gamma on `p` produces each
    /// continuation's leaf exactly once. No block is cloned.
    fn construct_loop_demux(
        &mut self,
        head: BasicBlockId,
        arms: &[LoopCaptureRegion],
        targets: &[DemuxBranchTarget],
        state: State,
        ctx: &LoopCtx,
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let arity = ctx.arity();

        // Capture layout: an in-region continuation captures its phis; a boundary
        // continuation captures its full leaf (computed in the head gamma,
        // forwarded by the demux gamma).
        let mut captures: Vec<TargetCapture> = Vec::with_capacity(targets.len());
        let mut next_offset = 0usize;
        for target in targets {
            let (phis, types) = if target.in_region_tail.is_some() {
                let phis =
                    phi_instructions_at(&self.fn_ctx.func.basic_blocks[target.block.0 as usize]);
                let types = self.convert_phi_types(&phis)?;
                (phis, types)
            } else {
                (SmallVec::new(), ctx.leaf_types.clone())
            };
            let offset = next_offset;
            next_offset += types.len();
            captures.push(TargetCapture {
                phis,
                types,
                offset,
            });
        }
        let captured_count = next_offset;

        // Live-ins over the head arms (bounded by the continuations) and the
        // in-region tails (bounded by the loop boundary).
        let arm_targets = arm_target_blocks(self.fn_ctx, head)?;
        let mut walk_boundary: SmallVec<[BasicBlockId; 8]> =
            targets.iter().map(|target| target.block).collect();
        walk_boundary.extend_from_slice(ctx.boundary());
        let mut walked: FxHashSet<BasicBlockId> = arm_targets
            .iter()
            .flat_map(|&arm| collect_walked_blocks(self.fn_ctx, arm, &walk_boundary))
            .collect();
        for target in targets {
            if target.in_region_tail.is_some() {
                walked.extend(collect_walked_blocks(
                    self.fn_ctx,
                    target.block,
                    ctx.boundary(),
                ));
            }
        }
        let walked_vec: Vec<BasicBlockId> = walked.into_iter().collect();
        let (live_in_names, live_ins) = self.loop_live_ins_over(&walked_vec, head, ctx);
        let predicate = self.branch_predicate(head)?;

        // ---- head gamma: discover `p`, capture per-continuation data ----------
        let fn_ctx = self.fn_ctx;
        let names_ref: &[Name] = &live_in_names;
        let captures_ref: &[TargetCapture] = &captures;
        let ctx_ref = ctx;
        let demux_cx = LoopDemuxCtx {
            targets,
            captures: captures_ref,
            captured_count,
            loop_ctx: ctx,
        };
        let demux_cx_ref = &demux_cx;

        let head_closures: Vec<_> = arms
            .iter()
            .map(|arm_region| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut arm = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                    arm.construct_loop_capture(arm_region, state, Some(head), demux_cx_ref)
                }
            })
            .collect();
        let head_refs = branch_refs(&head_closures);
        let outer = self.rb.gamma_n(predicate, state, &live_ins, &head_refs)?;

        let p_index = outer.result(0);
        let captured: Vec<ValueId> = (0..captured_count)
            .map(|i| outer.result(1 + i as u16))
            .collect();
        let p = self.rb.identity_match(p_index, targets.len() as u32);

        // ---- demux gamma: produce each continuation's leaf once ---------------
        let mut demux_inputs: Vec<ValueId> = live_ins.clone();
        let n_live = live_in_names.len();
        demux_inputs.extend_from_slice(&captured);

        let demux_closures: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(index, target)| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let base = n_live + captures_ref[index].offset;
                    match &target.in_region_tail {
                        None => {
                            // Boundary: the leaf is the captured block, forwarded.
                            let leaf: Vec<ValueId> = (0..arity)
                                .map(|slot| rb.param((base + slot) as u32))
                                .collect();
                            Ok(BranchResult {
                                state,
                                values: leaf,
                            })
                        }
                        Some(tail) => {
                            let mut ntv = FxHashMap::default();
                            seed_params(rb, names_ref, 0, &mut ntv);
                            for (slot, phi) in captures_ref[index].phis.iter().enumerate() {
                                ntv.insert(phi.dest.clone(), rb.param((base + slot) as u32));
                            }
                            let mut arm = RegionLowerer::new_child(rb, fn_ctx, ntv);
                            let (arm_state, leaf) =
                                arm.construct_loop_body(tail, state, None, ctx_ref)?;
                            Ok(BranchResult {
                                state: arm_state,
                                values: leaf,
                            })
                        }
                    }
                }
            })
            .collect();
        let demux_refs = branch_refs(&demux_closures);
        let demux = self
            .rb
            .gamma_n(p, outer.state, &demux_inputs, &demux_refs)?;
        let leaf: Vec<ValueId> = (0..arity).map(|i| demux.result(i as u16)).collect();
        Ok((demux.state, leaf))
    }

    /// Walk one head arm of a loop-body demux, producing its `[p, captures..]`
    /// leaf: `p` is the index of the continuation reached; then per continuation
    /// either its captured phis / full boundary leaf (for the reached one) or
    /// poison. A router arm emits a nested gamma whose arms recurse here.
    fn construct_loop_capture(
        &mut self,
        region: &LoopCaptureRegion,
        state: State,
        entry_prev: Option<BasicBlockId>,
        demux: &LoopDemuxCtx,
    ) -> color_eyre::Result<BranchResult> {
        let ctx = demux.loop_ctx;
        let mut walk_boundary: SmallVec<[BasicBlockId; 8]> =
            demux.targets.iter().map(|t| t.block).collect();
        walk_boundary.extend_from_slice(ctx.boundary());
        let (state, _prev) = self.lower_items(&region.items, state, entry_prev, &walk_boundary)?;

        match &region.exit {
            LoopCaptureExit::ToContinuation { reached, via } => {
                let reached_index = demux
                    .targets
                    .iter()
                    .position(|target| target.block == *reached)
                    .ok_or_else(|| {
                        color_eyre::eyre::eyre!(
                            "loop-demux arm reached {} which is not a continuation",
                            reached.0
                        )
                    })?;
                let mut values: Vec<ValueId> = Vec::with_capacity(1 + demux.captured_count);
                values.push(self.rb.const_i32(reached_index as i32));
                for (index, target) in demux.targets.iter().enumerate() {
                    if index == reached_index {
                        if target.in_region_tail.is_some() {
                            values.extend(
                                self.resolve_arm_join_phis(&demux.captures[index].phis, *via)?,
                            );
                        } else {
                            values.extend(self.loop_leaf(*reached, *via, ctx)?);
                        }
                    } else {
                        for &ty in &demux.captures[index].types {
                            values.push(self.rb.constant(ty, ConstValue::Poison));
                        }
                    }
                }
                Ok(BranchResult { state, values })
            }
            LoopCaptureExit::Route { head, arms } => {
                let arm_targets = arm_target_blocks(self.fn_ctx, *head)?;
                let walked: Vec<BasicBlockId> = arm_targets
                    .iter()
                    .flat_map(|&target| collect_walked_blocks(self.fn_ctx, target, &walk_boundary))
                    .collect();
                let (names, live_ins) = self.loop_live_ins_over(&walked, *head, ctx);
                let predicate = self.branch_predicate(*head)?;

                let fn_ctx = self.fn_ctx;
                let names_ref: &[Name] = &names;
                let demux_ref = demux;
                let route_head = *head;

                let sub_closures: Vec<_> = arms
                    .iter()
                    .map(|arm_region| {
                        move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                            let mut sub = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                            sub.construct_loop_capture(
                                arm_region,
                                state,
                                Some(route_head),
                                demux_ref,
                            )
                        }
                    })
                    .collect();
                let sub_refs = branch_refs(&sub_closures);
                let nested = self.rb.gamma_n(predicate, state, &live_ins, &sub_refs)?;
                let values: Vec<ValueId> = (0..(1 + demux.captured_count) as u16)
                    .map(|i| nested.result(i))
                    .collect();
                Ok(BranchResult {
                    state: nested.state,
                    values,
                })
            }
        }
    }

    /// The leaf for a loop body reaching `reached` from `via`: a repeat leaf if
    /// `reached` is a repeat boundary (the header / an entry vertex), an exit leaf
    /// if it is an exit vertex.
    fn loop_leaf(
        &mut self,
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
        ctx: &LoopCtx,
    ) -> color_eyre::Result<Vec<ValueId>> {
        let repeat_src = || {
            via.ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "repeat leaf reaching {} with no latch predecessor",
                    reached.0
                )
            })
        };
        let exit_src =
            || via.ok_or_else(|| color_eyre::eyre::eyre!("exit leaf with no exit predecessor"));
        match &ctx.kind {
            LoopKind::Reducible(reducible) => {
                if reached == reducible.header {
                    let src = repeat_src()?;
                    self.reducible_repeat_leaf(src, reducible)
                } else if let Some(exit_index) = reducible
                    .exit_blocks
                    .iter()
                    .position(|&block| block == reached)
                {
                    let src = exit_src()?;
                    self.reducible_exit_leaf(src, reached, exit_index, reducible)
                } else {
                    Err(color_eyre::eyre::eyre!(
                        "loop body reached {} which is not a loop boundary",
                        reached.0
                    ))
                }
            }
            LoopKind::MultiEntry(multi) => {
                if multi.entries.contains(&reached) {
                    let src = repeat_src()?;
                    self.multi_repeat_leaf(reached, src, multi)
                } else if multi.exit_targets.contains(&reached) {
                    self.multi_exit_leaf(reached, multi)
                } else {
                    Err(color_eyre::eyre::eyre!(
                        "irreducible loop body reached {} which is not a loop boundary",
                        reached.0
                    ))
                }
            }
        }
    }

    /// Single-entry repeat leaf for the back-edge from `src`: header phis take
    /// their incoming for `src` (per-latch), live-ins pass through, exit-phi and
    /// exit-`q` slots are poison, `r = 1`.
    fn reducible_repeat_leaf(
        &mut self,
        src: BasicBlockId,
        reducible: &ReducibleLoop,
    ) -> color_eyre::Result<Vec<ValueId>> {
        let mut leaf = Vec::with_capacity(reducible.arity());
        for phi in &reducible.header_phis {
            let (operand, _) = phi_incoming_from(phi, self.fn_ctx.bb_mapper, |block| block == src)
                .ok_or_else(|| {
                    color_eyre::eyre::eyre!(
                        "header phi {:?} has no incoming for latch {}",
                        phi.dest,
                        src.0
                    )
                })?;
            leaf.push(self.operand(operand)?);
        }
        for name in &reducible.live_in_names {
            leaf.push(self.resolve_loop_name(name)?);
        }
        for name in &reducible.closed {
            leaf.push(self.resolve_loop_name(name)?);
        }
        for (_, _, ty) in &reducible.exit_phis {
            leaf.push(self.rb.constant(*ty, ConstValue::Poison));
        }
        if reducible.has_exit_q {
            leaf.push(self.rb.constant(I32, ConstValue::Poison));
        }
        leaf.push(self.rb.constant(I32, ConstValue::Int(1)));
        Ok(leaf)
    }

    /// Single-entry exit leaf for arc `src -> vertex` (the `exit_index`-th exit
    /// vertex): header and live-in slots pass through, each exit phi at `vertex`
    /// resolves to its incoming for `src` (poison for other vertices' exit phis),
    /// the exit-`q` slot records `exit_index`, `r = 0`.
    fn reducible_exit_leaf(
        &mut self,
        src: BasicBlockId,
        vertex: BasicBlockId,
        exit_index: usize,
        reducible: &ReducibleLoop,
    ) -> color_eyre::Result<Vec<ValueId>> {
        let mut leaf = Vec::with_capacity(reducible.arity());
        for dest in &reducible.header_dests {
            leaf.push(self.resolve_loop_name(dest)?);
        }
        for name in &reducible.live_in_names {
            leaf.push(self.resolve_loop_name(name)?);
        }
        for name in &reducible.closed {
            leaf.push(self.resolve_loop_name(name)?);
        }
        for (phi_vertex, phi, ty) in &reducible.exit_phis {
            if *phi_vertex == vertex {
                let (operand, _) =
                    phi_incoming_from(phi, self.fn_ctx.bb_mapper, |block| block == src)
                        .ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "exit phi {:?} has no incoming for exit arc source {}",
                                phi.dest,
                                src.0
                            )
                        })?;
                leaf.push(self.operand(operand)?);
            } else {
                leaf.push(self.rb.constant(*ty, ConstValue::Poison));
            }
        }
        if reducible.has_exit_q {
            leaf.push(self.rb.constant(I32, ConstValue::Int(exit_index as i64)));
        }
        leaf.push(self.rb.constant(I32, ConstValue::Int(0)));
        Ok(leaf)
    }

    /// Multi-entry repeat leaf re-entering at `reentry` via arc from `src`: the
    /// re-entered vertex's phis take their incoming for `src`; every other entry
    /// vertex's phis and the closed values pass through; the entry `q` records the
    /// re-entered vertex; the exit `q` is unused; `r = 1`.
    fn multi_repeat_leaf(
        &mut self,
        reentry: BasicBlockId,
        src: BasicBlockId,
        multi: &MultiEntryLoop,
    ) -> color_eyre::Result<Vec<ValueId>> {
        let entry_index = multi
            .entries
            .iter()
            .position(|&entry| entry == reentry)
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("re-entry {} is not an entry vertex", reentry.0)
            })?;
        let mut leaf = Vec::with_capacity(multi.arity());
        for (entry, phi) in &multi.entry_phis {
            if *entry == reentry {
                let (operand, _) =
                    phi_incoming_from(phi, self.fn_ctx.bb_mapper, |block| block == src)
                        .ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "entry phi {:?} has no incoming for repetition arc from {}",
                                phi.dest,
                                src.0
                            )
                        })?;
                leaf.push(self.operand(operand)?);
            } else {
                leaf.push(self.resolve_loop_name(&phi.dest)?);
            }
        }
        for name in &multi.closed {
            leaf.push(self.resolve_loop_name(name)?);
        }
        leaf.push(self.rb.constant(I32, ConstValue::Int(entry_index as i64))); // entry q
        leaf.push(self.rb.constant(I32, ConstValue::Int(0))); // exit q (unused)
        leaf.push(self.rb.constant(I32, ConstValue::Int(1))); // r
        Ok(leaf)
    }

    /// Multi-entry exit leaf leaving via the arc into `target`: all entry phis and
    /// closed values pass through; the entry `q` is unused; the exit `q` records
    /// the exit target; `r = 0`.
    fn multi_exit_leaf(
        &mut self,
        target: BasicBlockId,
        multi: &MultiEntryLoop,
    ) -> color_eyre::Result<Vec<ValueId>> {
        let exit_index = multi
            .exit_targets
            .iter()
            .position(|&exit_target| exit_target == target)
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("exit target {} is not in exit targets", target.0)
            })?;
        let mut leaf = Vec::with_capacity(multi.arity());
        for (_, phi) in &multi.entry_phis {
            leaf.push(self.resolve_loop_name(&phi.dest)?);
        }
        for name in &multi.closed {
            leaf.push(self.resolve_loop_name(name)?);
        }
        leaf.push(self.rb.constant(I32, ConstValue::Int(0))); // entry q (unused)
        leaf.push(self.rb.constant(I32, ConstValue::Int(exit_index as i64))); // exit q
        leaf.push(self.rb.constant(I32, ConstValue::Int(0))); // r
        Ok(leaf)
    }

    /// Resolve a bound SSA name to its current value.
    fn resolve_loop_name(&self, name: &Name) -> color_eyre::Result<ValueId> {
        self.name_to_value
            .get(name)
            .copied()
            .ok_or_else(|| color_eyre::eyre::eyre!("loop leaf references unbound value {:?}", name))
    }

    /// Live-ins over an explicit walked block set for a loop-body construct,
    /// scanning the loop-variable phis and adding the loop-variable params that
    /// every leaf passes through.
    fn loop_live_ins_over(
        &self,
        walked: &[BasicBlockId],
        head: BasicBlockId,
        ctx: &LoopCtx,
    ) -> (Vec<Name>, Vec<ValueId>) {
        let (mut phis, pass_through): (Vec<&Phi>, Vec<Name>) = match &ctx.kind {
            LoopKind::Reducible(reducible) => {
                let mut phis = reducible.header_phis.clone();
                for (_, phi, _) in &reducible.exit_phis {
                    phis.push(phi);
                }
                let pass: Vec<Name> = reducible
                    .header_dests
                    .iter()
                    .chain(reducible.live_in_names.iter())
                    .chain(reducible.closed.iter())
                    .cloned()
                    .collect();
                (phis, pass)
            }
            LoopKind::MultiEntry(multi) => {
                let phis: Vec<&Phi> = multi.entry_phis.iter().map(|(_, phi)| *phi).collect();
                let pass: Vec<Name> = multi
                    .entry_phis
                    .iter()
                    .map(|(_, phi)| phi.dest.clone())
                    .chain(multi.closed.iter().cloned())
                    .collect();
                (phis, pass)
            }
        };
        phis.dedup_by(|left, right| std::ptr::eq(*left, *right));

        let (mut names, mut values) =
            region_live_ins(self.fn_ctx, &self.name_to_value, walked, &phis, Some(head));
        let mut seen: FxHashSet<Name> = names.iter().cloned().collect();
        for name in pass_through {
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(&value) = self.name_to_value.get(&name) {
                names.push(name);
                values.push(value);
            }
        }
        (names, values)
    }
}
