//! Lower conditional and switch terminators into gamma nodes. The
//! gamma's arms each own a region; arm blocks are identified by
//! dominator membership; each arm region's results match the join
//! block's phi shape.
//!
//! Bahmann, Reissmann, Jahre, Meyer (2015) section 4.2 (branch
//! restructuring). The build is driven by `restructure::continuation_points`
//! (Phase A): a branch with a single continuation point is a plain
//! single-join γ (`lower_n_way_branch`); a branch with more than one is
//! lowered by `lower_multi_continuation_branch` — the §4.2 `p`-demux — which
//! discovers the auxiliary continuation predicate `p` during the arm walk
//! and lowers each continuation exactly once (no node cloning).
//!
//! Not yet handled here: a branch with an arm that escapes the region
//! (returns, traps, or spins) instead of reaching a continuation point.
//! `lower_region`'s `arms_reconverge` guard detects this and falls back to
//! the single-join path (which clones the shared continuation but handles
//! the escaping arm); generalising the demux to escaping arms needs the
//! return/exit predicate plumbing.

use llvm_ir::{
    Name, Operand,
    instruction::Phi,
    terminator::{CondBr, Switch},
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::BasicBlockId,
        instructions::{RegionLowerer, for_each_operand, instruction_dest},
        region::{
            RegionExit,
            phi::{phi_incoming_from, phi_instructions_at},
        },
    },
    rvsdg::{
        ICmpPred, State, ValueId,
        builder::{BranchResult, RegionBuilder},
        types::{I32, TypeRef},
        value::ConstValue,
    },
};

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Lower a two-way conditional branch by delegating to the shared
    /// n-way gamma lowering with two arms.
    pub(super) fn lower_cond_branch(
        &mut self,
        state: State,
        cond_branch: &CondBr,
        head: BasicBlockId,
        join: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let predicate = self.operand(&cond_branch.condition)?;
        let true_target = *self.fn_ctx.bb_mapper.get_expect(&cond_branch.true_dest);
        let false_target = *self.fn_ctx.bb_mapper.get_expect(&cond_branch.false_dest);
        self.lower_n_way_branch(state, predicate, &[true_target, false_target], join, head)
    }

    /// Lower an n-way switch by building a gamma whose arm 0 is the
    /// default destination and arms `1..=N` are the case destinations.
    ///
    /// LLVM switches match arbitrary case values, but the gamma codegen
    /// (see `lower_to_llvm/gamma.rs`) emits an LLVM `switch cond, default
    /// [(1, b1), (2, b2), ...]` where arm `i` is selected when the
    /// condition equals `i`. To bridge that, we compute an "arm index"
    /// by chaining `icmp eq` + `select` over the case-value list: the
    /// index is `i+1` if the switch operand equals the i-th case value,
    /// else 0 (the default).
    pub(super) fn lower_switch(
        &mut self,
        state: State,
        switch: &Switch,
        head: BasicBlockId,
        join: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let (selector, targets) = self.switch_selector(switch)?;
        self.lower_n_way_branch(state, selector, &targets, join, head)
    }

    /// Compute the n-way gamma selector for a `switch`: arm 0 is the
    /// default destination and arm `i+1` is the i-th case, chosen by an
    /// `icmp eq` + `select` chain over the case values. Returns the
    /// selector value and the arm targets (default first, then cases in
    /// declaration order).
    ///
    /// Shared by the acyclic switch lowering above and the in-loop-body
    /// switch lowering (`loops::lower_body_switch`) — the two differ only
    /// in how the resulting (selector, targets) feed a gamma.
    pub(super) fn switch_selector(
        &mut self,
        switch: &Switch,
    ) -> color_eyre::Result<(ValueId, Vec<BasicBlockId>)> {
        let switch_operand = self.operand(&switch.operand)?;

        // Arm 0 is the default; arms 1..=N are case destinations in
        // declaration order.
        let mut targets = Vec::with_capacity(switch.dests.len() + 1);
        targets.push(*self.fn_ctx.bb_mapper.get_expect(&switch.default_dest));
        for (_, dest_name) in &switch.dests {
            targets.push(*self.fn_ctx.bb_mapper.get_expect(dest_name));
        }

        // Start at 0 (default) and for each case `i`, replace with `i+1`
        // when the switch operand matches that case's value.
        let mut selector = self.rb.const_i32(0);
        for (i, (case_const, _)) in switch.dests.iter().enumerate() {
            let case_value = self.operand(&Operand::ConstantOperand(case_const.clone()))?;
            let matched = self.rb.icmp(ICmpPred::Eq, switch_operand, case_value);
            let case_index = self.rb.const_i32((i + 1) as i32);
            selector = self.rb.ternary(matched, case_index, selector, I32);
        }

        Ok((selector, targets))
    }

    /// Shared body of `lower_cond_branch` and `lower_switch`: emit an
    /// n-arm gamma node and wire the join block's phis into the outer
    /// scope.
    ///
    /// For each arm target:
    ///   1. Compute the set of blocks the arm owns (dominated by the
    ///      target, not by the join).
    ///   2. Scan all arms together for SSA values used inside but
    ///      defined outside (the gamma's live-in inputs).
    ///   3. Build one closure per arm; each closure lowers its arm
    ///      region via `lower_arm`, which also resolves its
    ///      contribution to each join phi.
    ///   4. Emit the gamma via `gamma_n`, then bind each join-phi's
    ///      destination name to the corresponding gamma output.
    fn lower_n_way_branch(
        &mut self,
        state: State,
        predicate: ValueId,
        arm_targets: &[BasicBlockId],
        join: BasicBlockId,
        head: BasicBlockId,
    ) -> color_eyre::Result<State> {
        // The join block's phis define the gamma's per-arm result shape.
        let phis_at_join = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);

        // Per-arm block sets, owned so each closure can borrow into its
        // arm.
        let arm_blocks_per_arm: Vec<FxHashSet<BasicBlockId>> = arm_targets
            .iter()
            .map(|&target| self.collect_walked_blocks(target, &[join]))
            .collect();

        // Live-in scan needs every block across all arms. Flatten into
        // a SmallVec; arms are typically 1-3 blocks each.
        let combined_arm_blocks: SmallVec<[BasicBlockId; 8]> = arm_blocks_per_arm
            .iter()
            .flat_map(|set| set.iter().copied())
            .collect();
        let (live_in_names, live_ins) =
            self.compute_arm_live_ins(&combined_arm_blocks, &phis_at_join, Some(head));

        // Pre-bind into locals so closures don't capture `self`.
        let fn_ctx = self.fn_ctx;
        let phis_slice: &[&Phi] = &phis_at_join;
        let live_in_names_slice: &[Name] = &live_in_names;
        let boundary = [join];
        let boundary_ref: &[BasicBlockId] = &boundary;
        let leaf = Leaf::JoinPhis(phis_slice);
        let leaf_ref = &leaf;

        // Owned per-arm closures. Same closure type per iteration so
        // they can live together in a `Vec<_>`. Each walks its arm to the
        // join and produces its result slots via the leaf; `head` seeds the
        // walk so empty / shared-tail arms resolve correctly.
        let arm_closures: Vec<_> = arm_targets
            .iter()
            .map(|&target| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    lower_arm(
                        rb,
                        fn_ctx,
                        state,
                        target,
                        boundary_ref,
                        live_in_names_slice,
                        head,
                        leaf_ref,
                    )
                }
            })
            .collect();

        // Coerce each closure to a trait-object reference for gamma_n.
        let branch_refs = as_branch_refs(&arm_closures);
        let result = self.rb.gamma_n(predicate, state, &live_ins, &branch_refs)?;

        // Bind each join-phi's destination to the corresponding gamma
        // output so downstream uses resolve correctly.
        for (i, phi) in phis_at_join.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), result.result(i as u16));
        }

        Ok(result.state)
    }

    /// Lower a branch whose arms reconverge at MORE THAN ONE continuation
    /// point (Bahmann et al. §4.2, the `p`-demux case). Instead of cloning
    /// the shared continuation into every arm, we build two gammas:
    ///
    ///   1. `γ_outer` on the original predicate: each arm walks only as far
    ///      as the continuation-point boundary and reports WHICH one it
    ///      reached — the auxiliary predicate `p`. It outputs `p` followed
    ///      by, for each demux target, that target's phi values (resolved
    ///      from the arm that reached it; `poison` for targets it did not —
    ///      the demux never reads those slots).
    ///   2. `γ_demux` on `p`: one arm per demux target. Each binds its
    ///      target's phis from the captured `γ_outer` outputs, then lowers
    ///      the continuation from that target to the final `join` exactly
    ///      once. Its outputs are the `join`'s phi values.
    ///
    /// `join` (the post-dominator) is added to the demux targets so an arm
    /// that runs straight to the join is handled uniformly (an empty demux
    /// arm). On return the `join`'s phis are bound and the caller resumes
    /// there, exactly as for the single-join path.
    pub(super) fn lower_multi_continuation_branch(
        &mut self,
        state: State,
        predicate: ValueId,
        arm_targets: &[BasicBlockId],
        continuation_points: &[BasicBlockId],
        head: BasicBlockId,
        join: BasicBlockId,
    ) -> color_eyre::Result<State> {
        // Demux targets: the continuation points plus the final join.
        let mut demux_targets: SmallVec<[BasicBlockId; 4]> =
            continuation_points.iter().copied().collect();
        if !demux_targets.contains(&join) {
            demux_targets.push(join);
        }
        demux_targets.sort_unstable_by_key(|b| b.0);

        // Per demux target: its phi instructions and their RVSDG types.
        let target_phis: Vec<Vec<&Phi>> = demux_targets
            .iter()
            .map(|&t| {
                phi_instructions_at(&self.fn_ctx.func.basic_blocks[t.0 as usize])
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut target_phi_types: Vec<Vec<TypeRef>> = Vec::with_capacity(target_phis.len());
        for phis in &target_phis {
            let mut tys = Vec::with_capacity(phis.len());
            for phi in phis {
                tys.push(
                    self.rb
                        .graph
                        .types
                        .convert_type_ref(&phi.to_type, self.fn_ctx.llvm_mod)?,
                );
            }
            target_phi_types.push(tys);
        }

        // ---- Phase 1: γ_outer -------------------------------------------
        // Live-ins over the arms walked to the continuation-point boundary.
        let arm_blocks: Vec<FxHashSet<BasicBlockId>> = arm_targets
            .iter()
            .map(|&t| self.collect_walked_blocks(t, &demux_targets))
            .collect();
        let combined: SmallVec<[BasicBlockId; 8]> =
            arm_blocks.iter().flat_map(|s| s.iter().copied()).collect();
        let all_target_phis: Vec<&Phi> = target_phis.iter().flatten().copied().collect();
        let (live_in_names, live_ins) =
            self.compute_arm_live_ins(&combined, &all_target_phis, Some(head));

        let fn_ctx = self.fn_ctx;
        let demux_targets_slice: &[BasicBlockId] = &demux_targets;
        let target_phis_ref: &[Vec<&Phi>] = &target_phis;
        let target_phi_types_ref: &[Vec<TypeRef>] = &target_phi_types;
        let live_in_names_ref: &[Name] = &live_in_names;

        let outer_closures: Vec<_> = arm_targets
            .iter()
            .map(|&target| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut name_to_value = FxHashMap::default();
                    for (i, name) in live_in_names_ref.iter().enumerate() {
                        name_to_value.insert(name.clone(), rb.param(i as u32));
                    }
                    let mut arm = RegionLowerer::new_child(rb, fn_ctx, name_to_value);
                    let (arm_state, exit_pred, reached) = match arm.lower_region(
                        state,
                        target,
                        demux_targets_slice,
                        None,
                        Some(head),
                    )? {
                        RegionExit::AtBoundary {
                            state,
                            exit_pred,
                            reached,
                        } => (state, exit_pred, reached),
                        RegionExit::Returned { .. } => {
                            return Err(color_eyre::eyre::eyre!(
                                "early return inside a multi-continuation branch arm"
                            ));
                        }
                    };
                    let reached = reached.ok_or_else(|| {
                        color_eyre::eyre::eyre!(
                            "multi-continuation branch arm reached a dead end \
                             (unreachable) before any continuation point"
                        )
                    })?;
                    let k = demux_targets_slice
                        .iter()
                        .position(|&t| t == reached)
                        .ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "branch arm reached {} which is not a continuation point",
                                reached.0
                            )
                        })?;
                    // Output: p, then each target's phi values (the reached
                    // target resolved from this arm, the rest poison).
                    let mut values = Vec::new();
                    values.push(arm.rb.constant(I32, ConstValue::Int(k as i64)));
                    for (i, phis) in target_phis_ref.iter().enumerate() {
                        if i == k {
                            values.extend(arm.arm_phi_contributions(phis, exit_pred)?);
                        } else {
                            for &ty in &target_phi_types_ref[i] {
                                values.push(arm.rb.constant(ty, ConstValue::Poison));
                            }
                        }
                    }
                    Ok(BranchResult {
                        state: arm_state,
                        values,
                    })
                }
            })
            .collect();
        let outer_refs = as_branch_refs(&outer_closures);
        let outer = self.rb.gamma_n(predicate, state, &live_ins, &outer_refs)?;

        // Captured outputs: result(0) = p; result(1..) = per-target phis.
        let p = outer.result(0);
        // Offset (into the captured-phi inputs, i.e. excluding p) where each
        // target's phi block begins.
        let mut phi_offset: Vec<usize> = Vec::with_capacity(target_phis.len());
        let mut acc = 0usize;
        for phis in &target_phis {
            phi_offset.push(acc);
            acc += phis.len();
        }
        let captured_count = acc;
        let captured: Vec<ValueId> = (0..captured_count)
            .map(|i| outer.result(1 + i as u16))
            .collect();

        // ---- Phase 2: γ_demux -------------------------------------------
        // The final join's phis (the demux's uniform result shape).
        let join_phis = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);
        // Tail live-ins: outer-scope values used walking the demux targets to
        // the join. (Captured phis are bound per-arm, not live-ins.)
        let tail_blocks: Vec<FxHashSet<BasicBlockId>> = demux_targets
            .iter()
            .map(|&t| self.collect_walked_blocks(t, &[join]))
            .collect();
        let combined_tail: SmallVec<[BasicBlockId; 8]> =
            tail_blocks.iter().flat_map(|s| s.iter().copied()).collect();
        let (tail_names, tail_live_ins) =
            self.compute_arm_live_ins(&combined_tail, &join_phis, None);

        // γ_demux inputs: captured phis first, then the tail live-ins.
        let mut demux_inputs: Vec<ValueId> = captured.clone();
        demux_inputs.extend_from_slice(&tail_live_ins);

        let join_phis_ref: &[&Phi] = &join_phis;
        let tail_names_ref: &[Name] = &tail_names;
        let phi_offset_ref: &[usize] = &phi_offset;

        let demux_closures: Vec<_> = demux_targets
            .iter()
            .enumerate()
            .map(|(target_index, &target)| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut name_to_value = FxHashMap::default();
                    // Bind this target's phis from the captured γ_outer outputs.
                    let base = phi_offset_ref[target_index];
                    for (j, phi) in target_phis_ref[target_index].iter().enumerate() {
                        name_to_value.insert(phi.dest.clone(), rb.param((base + j) as u32));
                    }
                    // Bind the tail live-ins (params after the captured block).
                    for (j, name) in tail_names_ref.iter().enumerate() {
                        name_to_value.insert(name.clone(), rb.param((captured_count + j) as u32));
                    }
                    let mut arm = RegionLowerer::new_child(rb, fn_ctx, name_to_value);

                    let values = if target == join {
                        // Empty demux arm: already at the join; its phis are
                        // the ones just bound from the captured outputs.
                        arm.arm_phi_contributions(join_phis_ref, None)?
                    } else {
                        let (_, exit_pred) =
                            match arm.lower_region(state, target, &[join], None, None)? {
                                RegionExit::AtBoundary {
                                    state, exit_pred, ..
                                } => (state, exit_pred),
                                RegionExit::Returned { .. } => {
                                    return Err(color_eyre::eyre::eyre!(
                                        "early return inside a continuation-demux arm"
                                    ));
                                }
                            };
                        arm.arm_phi_contributions(join_phis_ref, exit_pred)?
                    };
                    // `BranchResult.state` is ignored by `gamma_n` (it only
                    // reads `values`); the arm's internal state edge is
                    // threaded by the backend, which serialises each region's
                    // nodes in insertion order, so a load/store the arm's tail
                    // walk emitted is ordered correctly within the arm. The
                    // `outer.state` here is a placeholder, consistent with the
                    // single-join path's `lower_arm`.
                    Ok(BranchResult {
                        state: outer.state,
                        values,
                    })
                }
            })
            .collect();
        let demux_refs = as_branch_refs(&demux_closures);
        let demux = self
            .rb
            .gamma_n(p, outer.state, &demux_inputs, &demux_refs)?;

        // Bind the join's phis from the demux outputs; the caller resumes
        // at the join.
        for (i, phi) in join_phis.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), demux.result(i as u16));
        }
        Ok(demux.state)
    }

    /// Scan a set of arm blocks to find SSA values that are USED inside
    /// but DEFINED outside. These become the gamma node's inputs
    /// (live-ins).
    ///
    /// Returns parallel vectors so the caller can correlate names with
    /// their outer `ValueId`s: `live_in_names[i]` is the LLVM Name;
    /// `live_ins[i]` is its resolution in the outer scope, both at the
    /// same index. The closures that build each arm's region will seed
    /// their `name_to_value` with `name -> arm_rb.param(i)`, mapping
    /// the outer `ValueId` to a region parameter.
    ///
    /// Two passes for clarity over fewer allocations:
    ///   1. Walk every instruction's *dest* to build `defined_inside`.
    ///   2. Walk every instruction's *operands* via `for_each_operand`;
    ///      any `LocalOperand` whose name isn't in `defined_inside` is
    ///      a candidate live-in. Resolve via `self.name_to_value`; skip
    ///      if unknown there (the caller will hit a clearer error
    ///      during arm lowering rather than us silently producing a
    ///      wrong gamma input).
    ///
    /// Phi operands at the join also need to be considered: each arm
    /// contributes one operand per phi, and that operand is resolved
    /// inside the arm's region (so its name must be in the arm's
    /// `name_to_value`). We scan phi operands whose predecessor block
    /// sits in *any* arm-block set; both arms share the unified live-in
    /// list so it's safe to merge.
    ///
    /// `defined_inside` and `seen` hold `&Name` references into the
    /// function's basic blocks, which outlive this call. No Name
    /// cloning happens until an entry is actually pushed onto the
    /// output Vec.
    ///
    /// For a switch with N arms, the caller flattens all arm-block sets
    /// into a single slice before calling; allocation cost is one short
    /// Vec per gamma.
    ///
    /// `pass_through_pred` is the head/branch block when this is a branch
    /// gamma (and `None` for the loop exit-dispatch path). An empty arm
    /// (a fan-out arc going straight to the join, e.g. the `else` of a
    /// short-circuit `&&`/`||`) contributes its join-phi value along the
    /// `head -> join` edge. That value is defined in the outer scope, so
    /// it must be threaded in as a live-in here; the arm then echoes it
    /// straight back out as its result.
    pub(super) fn compute_arm_live_ins(
        &self,
        arm_block_set: &[BasicBlockId],
        phis_at_join: &[&Phi],
        pass_through_pred: Option<BasicBlockId>,
    ) -> (Vec<Name>, Vec<ValueId>) {
        // Pass 1: names defined inside the arms.
        let mut defined_inside: FxHashSet<&Name> = FxHashSet::default();
        for &bb_id in arm_block_set {
            let bb = &self.fn_ctx.func.basic_blocks[bb_id.0 as usize];
            for inst in &bb.instrs {
                if let Some(dest) = instruction_dest(inst) {
                    defined_inside.insert(dest);
                }
            }
        }

        // Pass 2: operands used inside the arms but not defined there.
        let mut seen: FxHashSet<&Name> = FxHashSet::default();
        let mut names: Vec<Name> = Vec::new();
        let mut values: Vec<ValueId> = Vec::new();

        for &bb_id in arm_block_set {
            let bb = &self.fn_ctx.func.basic_blocks[bb_id.0 as usize];
            for inst in &bb.instrs {
                for_each_operand(inst, |op| {
                    let Operand::LocalOperand { name, .. } = op else {
                        return;
                    };
                    if defined_inside.contains(name) || !seen.insert(name) {
                        return;
                    }
                    if let Some(&val) = self.name_to_value.get(name) {
                        names.push(name.clone());
                        values.push(val);
                    }
                });
            }
        }

        // Phi operands at the join: each arm contributes one per phi.
        // The arm's closure resolves that operand inside the arm
        // region, so the name must be in the arm's `name_to_value`,
        // which we seed from live-ins. Membership is a linear scan on
        // `arm_block_set`; arms are typically 1-8 blocks, so the slice
        // scan beats allocating a hash set just for this check.
        for phi in phis_at_join {
            for (op, pred_name) in &phi.incoming_values {
                let Some(&pred_id) = self.fn_ctx.bb_mapper.get(pred_name) else {
                    continue;
                };
                // Accept operands flowing in from an arm block (normal
                // arms) or directly along the head -> join edge (the
                // empty pass-through arm).
                if !arm_block_set.contains(&pred_id) && Some(pred_id) != pass_through_pred {
                    continue;
                }
                let Operand::LocalOperand { name, .. } = op else {
                    continue;
                };
                if defined_inside.contains(name) || !seen.insert(name) {
                    continue;
                }
                if let Some(&val) = self.name_to_value.get(name) {
                    names.push(name.clone());
                    values.push(val);
                }
            }
        }

        (names, values)
    }

    /// The set of blocks a gamma arm actually *walks*: every block
    /// reachable from `arm_root` by following CFG successors, stopping at
    /// (and not crossing) `boundary` and the synthetic function exit.
    ///
    /// This is the region the arm's `lower_region` covers, and per the
    /// paper's BUILD_RVSDG* (§4, lines 458-461: a gamma takes "all
    /// variables required in the subregions" as inputs) it is exactly the
    /// set whose SSA uses-minus-defs are the arm's live-ins.
    ///
    /// It deliberately differs from the dominator set used previously: a
    /// continuation block reached from this arm — a switch fall-through
    /// tail, a cross edge — is dominated by neither arm yet IS walked, so
    /// its outer-scope uses must be threaded in as gamma inputs. Scanning
    /// only the dominator set omitted those, which is the use-before-def
    /// defect (`instructions.rs:898`). Because `join` post-dominates
    /// `arm_root` in a well-formed branch, this reachability is exactly the
    /// arm region; SSA dominance then guarantees a value defined inside the
    /// region is defined before any use, so the caller's uses-minus-defs
    /// scan is precise.
    ///
    /// Reachability is bounded by the region (typically 1-8 blocks),
    /// cheaper than the previous full-function dominator scan.
    pub(super) fn collect_walked_blocks(
        &self,
        arm_root: BasicBlockId,
        boundary: &[BasicBlockId],
    ) -> FxHashSet<BasicBlockId> {
        let mut set = FxHashSet::default();
        let exit = self.fn_ctx.exit_block_id;
        let stops = |b: BasicBlockId| b == exit || boundary.contains(&b);
        // An empty arm (fan-out arc straight to a boundary) walks nothing.
        let mut stack: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
        if !stops(arm_root) {
            stack.push(arm_root);
        }
        while let Some(bb) = stack.pop() {
            if !set.insert(bb) {
                continue;
            }
            for &succ in self.fn_ctx.bb_mapper.outputs(bb) {
                if !stops(succ) && !set.contains(&succ) {
                    stack.push(succ);
                }
            }
        }
        set
    }

    /// For each phi at the join, resolve the value *this* arm contributes,
    /// in the arm's region scope.
    ///
    /// Every arm must produce exactly one value per join phi. The value is
    /// found, in order:
    ///
    /// 1. If the arm already bound the phi's destination — an inner gamma
    ///    whose join coincides with this one (e.g. a nested if/else whose
    ///    merge is the same block) leaves the resolved value in
    ///    `name_to_value`. Use it directly.
    /// 2. Otherwise the arm reached the join linearly from `exit_pred`
    ///    (the block its walk last stepped through, reported by
    ///    `lower_region`). Pick the phi incoming for that predecessor and
    ///    resolve it in the arm's scope. This covers the normal single
    ///    block arm, an empty pass-through arm (where `exit_pred` is the
    ///    branching head), and a shared continuation block reached via a
    ///    fall-through (where `exit_pred` is that shared block).
    ///
    /// Errors if neither resolves — a malformed phi or a shape that still
    /// needs the not-yet-built multi-continuation `p` dispatch.
    pub(super) fn arm_phi_contributions(
        &mut self,
        phis_at_join: &[&Phi],
        exit_pred: Option<BasicBlockId>,
    ) -> color_eyre::Result<Vec<ValueId>> {
        phis_at_join
            .iter()
            .map(|phi| {
                // 1. Already bound by an inner gamma sharing this join.
                if let Some(&v) = self.name_to_value.get(&phi.dest) {
                    return Ok(v);
                }
                // 2. The incoming for the predecessor the arm exited from.
                if let Some(pred) = exit_pred {
                    if let Some((operand, _)) =
                        phi_incoming_from(phi, self.fn_ctx.bb_mapper, |id| id == pred)
                    {
                        return self.operand(operand);
                    }
                }
                Err(color_eyre::eyre::eyre!(
                    "phi {:?} at join has no incoming value from this arm \
                     (exit predecessor: {:?}, predecessors in phi: {:?})",
                    phi.dest,
                    exit_pred,
                    phi.incoming_values
                        .iter()
                        .map(|(_, p)| p)
                        .collect::<Vec<_>>(),
                ))
            })
            .collect()
    }
}

/// Builds a fresh `RegionLowerer` whose `name_to_value` is seeded with
/// the arm's region parameters (one per live-in, in `live_in_names`
/// order), lowers the arm's blocks up to but not including `join`,
/// then resolves each join-phi's contribution from this arm via
/// `arm_phi_contributions`.
///
/// A `Returned` exit from the arm region means an arm hit `Ret` before
/// reaching the join. That early-return shape isn't supported yet; we
/// surface it as an error rather than silently dropping return values.
///
/// Free function rather than a method because the closure that
/// `gamma_n` calls passes a different `RegionBuilder` (the arm's
/// region builder, not the caller's). The arm's `RegionLowerer` is
/// built inside this function from that fresh builder.
fn lower_arm(
    rb: &mut RegionBuilder,
    fn_ctx: &FnCtx,
    state: State,
    arm_root: BasicBlockId,
    boundary: &[BasicBlockId],
    live_in_names: &[Name],
    head: BasicBlockId,
    leaf: &Leaf,
) -> color_eyre::Result<BranchResult> {
    let mut name_to_value = FxHashMap::default();
    for (i, name) in live_in_names.iter().enumerate() {
        name_to_value.insert(name.clone(), rb.param(i as u32));
    }

    let mut arm = RegionLowerer::new_child(rb, fn_ctx, name_to_value);

    // Seed the walk's predecessor with the branching head so that an
    // empty arm (arm_root in `boundary`) reports the head as its exit
    // predecessor, and a shared first block resolves its phis path-aware.
    let (state, exit_pred, reached) =
        match arm.lower_region(state, arm_root, boundary, None, Some(head))? {
            RegionExit::AtBoundary {
                state,
                exit_pred,
                reached,
            } => (state, exit_pred, reached),
            RegionExit::Returned { .. } => {
                return Err(color_eyre::eyre::eyre!(
                    "early returns are not supported within gamma-arm lowering"
                ));
            }
        };

    let values = leaf.produce(&mut arm, reached, exit_pred)?;

    Ok(BranchResult { state, values })
}

/// What a gamma arm produces when its walk reaches a region boundary — the
/// arm's per-result-slot values. This is the one knob that lets the
/// single-join γ, the `p`-demux, and the loop body share a single lowering
/// primitive while differing only in the terminal computation.
pub(super) enum Leaf<'a> {
    /// Acyclic branch: the result slots are the join block's phis, resolved
    /// from the predecessor the arm exited through.
    JoinPhis(&'a [&'a Phi]),
}

impl Leaf<'_> {
    /// Produce the arm's result vector for a path that reached boundary
    /// block `reached` (via predecessor `exit_pred`).
    fn produce(
        &self,
        arm: &mut RegionLowerer,
        _reached: Option<BasicBlockId>,
        exit_pred: Option<BasicBlockId>,
    ) -> color_eyre::Result<Vec<ValueId>> {
        match self {
            Leaf::JoinPhis(phis) => arm.arm_phi_contributions(phis, exit_pred),
        }
    }
}

/// Coerce a slice of owned branch closures into the `&dyn Fn` trait-object
/// references `RegionBuilder::gamma_n` expects. Centralises the coercion
/// repeated at every gamma-construction site.
pub(super) fn as_branch_refs<F>(
    closures: &[F],
) -> Vec<&dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>>
where
    F: Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>,
{
    closures
        .iter()
        .map(|c| c as &dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::region::{
        phi::phi_instructions_at,
        test_fixture::{TestFn, local_name},
    };
    use pretty_assertions::assert_eq;

    // ------------------------------------------------------------------------
    // collect_walked_blocks
    // ------------------------------------------------------------------------

    #[test]
    fn collect_walked_blocks_simple_diamond() {
        // entry -> t -> j
        //       -> f -> j
        // Each arm is a single block; the walk stops at the join.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %t ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let true_arm = lowerer.collect_walked_blocks(t, &[j]);
            assert_eq!(true_arm.len(), 1);
            assert!(true_arm.contains(&t));
            assert!(!true_arm.contains(&j), "join must not be in the arm");

            let false_arm = lowerer.collect_walked_blocks(f, &[j]);
            assert_eq!(false_arm.len(), 1);
            assert!(false_arm.contains(&f));
        });
    }

    #[test]
    fn collect_walked_blocks_includes_chain() {
        // entry -> t -> mid -> j
        //       -> f -> j
        // The true arm spans both `t` and `mid` (reachable before the join).
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %mid
mid:
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %mid ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let mid = test_fn.block("mid");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let true_arm = lowerer.collect_walked_blocks(t, &[j]);
            assert_eq!(true_arm.len(), 2);
            assert!(true_arm.contains(&t));
            assert!(true_arm.contains(&mid));
            assert!(!true_arm.contains(&j), "join must not be in the arm");
            assert!(!true_arm.contains(&f), "other arm must not be in the arm");
        });
    }

    #[test]
    fn collect_walked_blocks_excludes_blocks_past_join() {
        // entry -> t -> j -> after
        //       -> f -> j
        // `after` is past the join; the walk stops at the join.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %t ], [ %b, %f ]
  br label %after
after:
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let j = test_fn.block("j");
        let after = test_fn.block("after");

        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let true_arm = lowerer.collect_walked_blocks(t, &[j]);
            assert!(
                !true_arm.contains(&after),
                "post-join block must not be in arm"
            );
        });
    }

    #[test]
    fn collect_walked_blocks_includes_shared_continuation() {
        // entry -> t ------> shared -> j
        //       -> f -> shared
        // `shared` is reached from BOTH arms, so it is dominated by neither
        // (its idom is `entry`). The old dominator-set collection excluded
        // it from both arms even though both walk it — the use-before-def
        // bug. Reachability must include it in each arm that reaches it.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %shared
f:
  br label %shared
shared:
  %s = phi i32 [ %a, %t ], [ %b, %f ]
  br label %j
j:
  ret i32 %s
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let shared = test_fn.block("shared");
        let j = test_fn.block("j");

        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let true_arm = lowerer.collect_walked_blocks(t, &[j]);
            assert!(true_arm.contains(&t));
            assert!(
                true_arm.contains(&shared),
                "shared continuation reached from this arm must be walked"
            );

            let false_arm = lowerer.collect_walked_blocks(f, &[j]);
            assert!(false_arm.contains(&f));
            assert!(
                false_arm.contains(&shared),
                "shared continuation reached from this arm must be walked"
            );
        });
    }

    #[test]
    fn collect_walked_blocks_empty_arm_is_empty() {
        // entry -> j (true arm goes straight to the join: an empty arm)
        //       -> f -> j
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %j, label %f
f:
  br label %j
j:
  %r = phi i32 [ %a, %entry ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        let j = test_fn.block("j");

        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let empty_arm = lowerer.collect_walked_blocks(j, &[j]);
            assert!(
                empty_arm.is_empty(),
                "an arm that is the join walks nothing"
            );
        });
    }

    // ------------------------------------------------------------------------
    // compute_arm_live_ins
    // ------------------------------------------------------------------------

    #[test]
    fn live_ins_include_phi_operands_from_function_params() {
        // The phi at the join references function params %a and %b;
        // both must be picked up as live-ins (this is the regression
        // that caused the earlier crash on the `max` example).
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %t ], [ %b, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        let arm_blocks = vec![t, f];

        let mut name_to_value = FxHashMap::default();
        name_to_value.insert(local_name("a"), ValueId(100));
        name_to_value.insert(local_name("b"), ValueId(101));

        test_fn.with_lowerer(name_to_value, |lowerer| {
            let (names, values) = lowerer.compute_arm_live_ins(&arm_blocks, &phis, None);
            assert!(names.contains(&local_name("a")), "%a should be a live-in");
            assert!(names.contains(&local_name("b")), "%b should be a live-in");
            assert_eq!(names.len(), values.len());
            assert_eq!(names.len(), 2);
        });
    }

    #[test]
    fn live_ins_skip_locally_defined_names() {
        // %local and %used_local are both defined inside the true arm;
        // neither should be flagged as a live-in even though they're
        // used inside.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  %local = add i32 %a, 1
  %used_local = add i32 %local, 1
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %used_local, %t ], [ %a, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        let arm_blocks = vec![t, f];

        let mut name_to_value = FxHashMap::default();
        name_to_value.insert(local_name("a"), ValueId(100));

        test_fn.with_lowerer(name_to_value, |lowerer| {
            let (names, _) = lowerer.compute_arm_live_ins(&arm_blocks, &phis, None);
            assert!(
                names.contains(&local_name("a")),
                "param %a must be a live-in"
            );
            assert!(
                !names.contains(&local_name("local")),
                "%local defined inside arm must not be a live-in"
            );
            assert!(
                !names.contains(&local_name("used_local")),
                "%used_local defined inside arm must not be a live-in"
            );
        });
    }

    #[test]
    fn live_ins_skip_constant_operands() {
        // Operands `1` and `2` are constants; only the local %a should
        // be picked up.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  %x = add i32 1, 2
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %x, %t ], [ %a, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        let arm_blocks = vec![t, f];

        let mut name_to_value = FxHashMap::default();
        name_to_value.insert(local_name("a"), ValueId(100));

        test_fn.with_lowerer(name_to_value, |lowerer| {
            let (names, values) = lowerer.compute_arm_live_ins(&arm_blocks, &phis, None);
            assert_eq!(names, vec![local_name("a")]);
            assert_eq!(values, vec![ValueId(100)]);
        });
    }

    #[test]
    fn live_ins_silently_skip_unresolved_names() {
        // outer_map is empty; %a is referenced but unresolvable. We
        // must not panic; we leave the missing-value error for the arm
        // lowerer to surface at the actual use site.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %t ], [ %a, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        let arm_blocks = vec![t, f];

        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let (names, values) = lowerer.compute_arm_live_ins(&arm_blocks, &phis, None);
            assert!(names.is_empty());
            assert!(values.is_empty());
        });
    }

    #[test]
    fn live_ins_pick_up_operands_from_arm_instructions() {
        // %x is used inside the arm but defined outside (we pretend it
        // lives in the outer scope via outer_map). It must surface as a
        // live-in even though no phi at the join mentions it.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i32 %x, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  %y = add i32 %x, %x
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %y, %t ], [ %a, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        let arm_blocks = vec![t, f];

        let mut name_to_value = FxHashMap::default();
        name_to_value.insert(local_name("a"), ValueId(100));
        name_to_value.insert(local_name("x"), ValueId(200));

        test_fn.with_lowerer(name_to_value, |lowerer| {
            let (names, _) = lowerer.compute_arm_live_ins(&arm_blocks, &phis, None);
            assert!(
                names.contains(&local_name("x")),
                "%x used inside arm is a live-in"
            );
            assert!(
                names.contains(&local_name("a")),
                "%a from phi op is a live-in"
            );
            assert!(
                !names.contains(&local_name("y")),
                "%y defined inside arm is not a live-in"
            );
        });
    }

    #[test]
    fn live_ins_deduplicate_repeated_uses() {
        // %a used twice in t and in the phi; should appear only once
        // in the live-in list.
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  %y = add i32 %a, %a
  br label %j
f:
  br label %j
j:
  %r = phi i32 [ %a, %t ], [ %a, %f ]
  ret i32 %r
}
"#,
        );
        let t = test_fn.block("t");
        let f = test_fn.block("f");
        let j = test_fn.block("j");

        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        let arm_blocks = vec![t, f];

        let mut name_to_value = FxHashMap::default();
        name_to_value.insert(local_name("a"), ValueId(100));

        test_fn.with_lowerer(name_to_value, |lowerer| {
            let (names, _) = lowerer.compute_arm_live_ins(&arm_blocks, &phis, None);
            assert_eq!(
                names.iter().filter(|n| **n == local_name("a")).count(),
                1,
                "%a should appear once even though referenced multiple times"
            );
        });
    }
}
