//! Construction of branch (gamma) nodes: split-join gammas, the two-gamma
//! continuation-demux (and its recursive capture arms), the non-reconverging
//! all-arms-return gamma, and the per-branch control predicate.

use llvm_ir::{Name, instruction::Phi};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        block_mapper::BasicBlockId,
        control_flow::{
            analysis::signature::{collect_walked_blocks, phi_instructions_at, region_live_ins},
            construct::{ConstructExit, TargetCapture, branch_refs, seed_params},
            restructure::arm_target_blocks,
            rst::{
                CaptureExit, CaptureRegion, DemuxSpec, DemuxTail, GammaMerge, GammaNode, SeqRegion,
            },
        },
        instructions::RegionLowerer,
    },
    rvsdg::{
        State, ValueId,
        builder::{BranchResult, RegionBuilder},
        types::{I32, TypeRef, VOID},
        value::ConstValue,
    },
};

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Emit a gamma node, returning the post-gamma state.
    pub(in crate::llvm_parser) fn construct_gamma(
        &mut self,
        gamma: &GammaNode,
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        match &gamma.merge {
            GammaMerge::SingleJoin { join, arms } => {
                self.construct_single_join_gamma(gamma.head, arms, *join, state, boundary)
            }
            GammaMerge::Demux { head_arms, spec } => {
                self.construct_demux_gamma(gamma.head, head_arms, spec, state, boundary)
            }
        }
    }

    /// Turn an arm's [`ConstructExit`] into a [`BranchResult`] that reconverges at
    /// `join`: an arm reaching `join` binds the join phis along its exit edge; a
    /// diverging arm fills poison of each join-phi type. Any other exit is an
    /// error tagged with `arm_kind` (e.g. `"single-join"`, `"exit-demux"`).
    pub(in crate::llvm_parser) fn join_arm_result(
        &mut self,
        exit: ConstructExit,
        join: BasicBlockId,
        join_phis: &[&Phi],
        phi_types: &[TypeRef],
        arm_kind: &str,
    ) -> color_eyre::Result<BranchResult> {
        match exit {
            ConstructExit::AtBoundary {
                state,
                reached,
                exit_pred,
            } if reached == join => {
                let values = self.resolve_arm_join_phis(join_phis, exit_pred)?;
                Ok(BranchResult { state, values })
            }
            ConstructExit::Diverge { state } => {
                let values = phi_types
                    .iter()
                    .map(|&ty| self.rb.constant(ty, ConstValue::Poison))
                    .collect();
                Ok(BranchResult { state, values })
            }
            ConstructExit::AtBoundary { reached, .. } => Err(color_eyre::eyre::eyre!(
                "{} arm reached {} (expected join {})",
                arm_kind,
                reached.0,
                join.0
            )),
            ConstructExit::Returned { .. } => Err(color_eyre::eyre::eyre!(
                "{} arm returned but the branch reconverges at block {}",
                arm_kind,
                join.0
            )),
        }
    }

    /// Emit a split-join gamma whose arms reconverge at `join`. The gamma outputs
    /// are the join phis, bound here so the enclosing region resumes at `join`.
    fn construct_single_join_gamma(
        &mut self,
        head: BasicBlockId,
        arms: &[SeqRegion],
        join: BasicBlockId,
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        let arm_targets = arm_target_blocks(self.fn_ctx, head)?;
        let join_phis = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);

        let mut arm_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
        arm_boundary.push(join);
        arm_boundary.extend_from_slice(boundary);

        let (live_in_names, live_ins) =
            self.live_ins_for_arms(&arm_targets, &arm_boundary, &join_phis, Some(head));
        let phi_types = self.convert_phi_types(&join_phis)?;
        let predicate = self.branch_predicate(head)?;

        let fn_ctx = self.fn_ctx;
        let names_ref: &[Name] = &live_in_names;
        let join_phis_ref: &[&Phi] = &join_phis;
        let phi_types_ref: &[TypeRef] = &phi_types;
        let arm_boundary_ref: &[BasicBlockId] = &arm_boundary;

        let arm_closures: Vec<_> = arms
            .iter()
            .map(|arm_region| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut arm = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                    let exit = arm.construct(arm_region, state, Some(head), arm_boundary_ref)?;
                    arm.join_arm_result(exit, join, join_phis_ref, phi_types_ref, "single-join")
                }
            })
            .collect();

        let refs = branch_refs(&arm_closures);
        let result = self.rb.gamma_n(predicate, state, &live_ins, &refs)?;
        for (i, phi) in join_phis.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), result.result(i as u16));
        }
        Ok(result.state)
    }

    /// Emit the two-gamma continuation-demux: a head gamma whose arms produce the
    /// demux predicate index plus each target's captured phis, then a demux gamma
    /// on that index that lowers each target's tail to `join` exactly once.
    fn construct_demux_gamma(
        &mut self,
        head: BasicBlockId,
        head_arms: &[CaptureRegion],
        spec: &DemuxSpec,
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<State> {
        let demux_targets = &spec.demux_targets;
        let join = spec.join;

        let mut captures: Vec<TargetCapture> = Vec::with_capacity(demux_targets.len());
        let mut next_offset = 0usize;
        for &target in demux_targets {
            let phis = phi_instructions_at(&self.fn_ctx.func.basic_blocks[target.0 as usize]);
            let types = self.convert_phi_types(&phis)?;
            let offset = next_offset;
            next_offset += types.len();
            captures.push(TargetCapture {
                phis,
                types,
                offset,
            });
        }
        let captured_count = next_offset;

        // ---- head gamma: discover the index `p` and capture per-target phis ----
        let all_target_phis: Vec<&Phi> = captures
            .iter()
            .flat_map(|capture| capture.phis.iter().copied())
            .collect();
        let arm_targets = arm_target_blocks(self.fn_ctx, head)?;
        let mut head_boundary: SmallVec<[BasicBlockId; 8]> =
            demux_targets.iter().copied().collect();
        head_boundary.extend_from_slice(boundary);
        let (head_names, head_live_ins) =
            self.live_ins_for_arms(&arm_targets, &head_boundary, &all_target_phis, Some(head));
        let predicate = self.branch_predicate(head)?;

        let fn_ctx = self.fn_ctx;
        let head_names_ref: &[Name] = &head_names;
        let demux_targets_ref: &[BasicBlockId] = demux_targets;
        let captures_ref: &[TargetCapture] = &captures;

        let head_closures: Vec<_> = head_arms
            .iter()
            .map(|arm_region| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut arm = RegionLowerer::arm_child(rb, fn_ctx, head_names_ref);
                    arm.construct_capture(
                        arm_region,
                        state,
                        Some(head),
                        demux_targets_ref,
                        captures_ref,
                        captured_count,
                    )
                }
            })
            .collect();
        let head_refs = branch_refs(&head_closures);
        let outer = self
            .rb
            .gamma_n(predicate, state, &head_live_ins, &head_refs)?;

        let p_index = outer.result(0);
        let captured: Vec<ValueId> = (0..captured_count)
            .map(|i| outer.result(1 + i as u16))
            .collect();
        let p = self.rb.identity_match(p_index, demux_targets.len() as u32);

        // ---- demux gamma: lower each target's tail to `join` exactly once ------
        let join_phis = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);
        let mut demux_boundary: SmallVec<[BasicBlockId; 8]> = SmallVec::new();
        demux_boundary.push(join);
        demux_boundary.extend_from_slice(boundary);
        let tail_walked: FxHashSet<BasicBlockId> = demux_targets
            .iter()
            .filter(|&&target| target != join)
            .flat_map(|&target| {
                let arm_boundary: SmallVec<[BasicBlockId; 8]> = demux_boundary
                    .iter()
                    .copied()
                    .filter(|&block| block != target)
                    .collect();
                collect_walked_blocks(self.fn_ctx, target, &arm_boundary)
            })
            .collect();
        let tail_walked_vec: Vec<BasicBlockId> = tail_walked.into_iter().collect();
        let (tail_names, tail_live_ins) = region_live_ins(
            self.fn_ctx,
            &self.name_to_value,
            &tail_walked_vec,
            &join_phis,
            None,
        );

        let mut demux_inputs: Vec<ValueId> = captured;
        demux_inputs.extend_from_slice(&tail_live_ins);

        let join_phis_ref: &[&Phi] = &join_phis;
        let tail_names_ref: &[Name] = &tail_names;
        let demux_boundary_ref: &[BasicBlockId] = &demux_boundary;
        let tails_ref: &[DemuxTail] = &spec.tails;

        let demux_closures: Vec<_> = demux_targets
            .iter()
            .enumerate()
            .map(|(target_index, &target)| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut ntv = FxHashMap::default();
                    let base = captures_ref[target_index].offset;
                    for (slot, phi) in captures_ref[target_index].phis.iter().enumerate() {
                        ntv.insert(phi.dest.clone(), rb.param((base + slot) as u32));
                    }
                    seed_params(rb, tail_names_ref, captured_count as u32, &mut ntv);
                    let mut arm = RegionLowerer::new_child(rb, fn_ctx, ntv);

                    let values = match &tails_ref[target_index] {
                        DemuxTail::Join => arm.resolve_arm_join_phis(join_phis_ref, None)?,
                        DemuxTail::Tail(tail) => {
                            match arm.construct(tail, state, None, demux_boundary_ref)? {
                                ConstructExit::AtBoundary {
                                    exit_pred, reached, ..
                                } if reached == join => {
                                    arm.resolve_arm_join_phis(join_phis_ref, exit_pred)?
                                }
                                ConstructExit::AtBoundary { reached, .. } => {
                                    return Err(color_eyre::eyre::eyre!(
                                        "demux tail from {} reached {} (expected join {})",
                                        target.0,
                                        reached.0,
                                        join.0
                                    ));
                                }
                                ConstructExit::Diverge { .. } => {
                                    return Err(color_eyre::eyre::eyre!(
                                        "demux tail from {} diverged before the join",
                                        target.0
                                    ));
                                }
                                ConstructExit::Returned { .. } => {
                                    return Err(color_eyre::eyre::eyre!(
                                        "early return inside a continuation-demux tail"
                                    ));
                                }
                            }
                        }
                    };
                    Ok(BranchResult {
                        state: outer.state,
                        values,
                    })
                }
            })
            .collect();
        let demux_refs = branch_refs(&demux_closures);
        let demux = self
            .rb
            .gamma_n(p, outer.state, &demux_inputs, &demux_refs)?;

        for (i, phi) in join_phis.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), demux.result(i as u16));
        }
        Ok(demux.state)
    }

    /// Construct a head-demux arm `region`, producing its `[p, captures..]` leaf:
    /// the index of the demux target it reaches, then each target's captured phis
    /// (poison for targets not reached). A region that ends in a router
    /// ([`CaptureExit::Route`]) emits a nested gamma whose arms recurse here.
    fn construct_capture(
        &mut self,
        region: &CaptureRegion,
        state: State,
        entry_prev: Option<BasicBlockId>,
        demux_targets: &[BasicBlockId],
        captures: &[TargetCapture],
        captured_count: usize,
    ) -> color_eyre::Result<BranchResult> {
        let (state, _prev) = self.lower_items(&region.items, state, entry_prev, demux_targets)?;

        match &region.exit {
            CaptureExit::ToContinuation { reached, via } => {
                let values =
                    self.demux_capture_leaf(Some(*reached), *via, demux_targets, captures)?;
                Ok(BranchResult { state, values })
            }
            CaptureExit::Diverge => {
                let values = self.demux_capture_leaf(None, None, demux_targets, captures)?;
                Ok(BranchResult { state, values })
            }
            CaptureExit::Route { head, arms } => {
                let all_target_phis: Vec<&Phi> = captures
                    .iter()
                    .flat_map(|capture| capture.phis.iter().copied())
                    .collect();
                let arm_targets = arm_target_blocks(self.fn_ctx, *head)?;
                let (names, live_ins) = self.live_ins_for_arms(
                    &arm_targets,
                    demux_targets,
                    &all_target_phis,
                    Some(*head),
                );
                let predicate = self.branch_predicate(*head)?;

                let fn_ctx = self.fn_ctx;
                let names_ref: &[Name] = &names;
                let demux_targets_ref = demux_targets;
                let captures_ref = captures;
                let route_head = *head;

                let sub_closures: Vec<_> = arms
                    .iter()
                    .map(|sub_region| {
                        move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                            let mut sub = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                            sub.construct_capture(
                                sub_region,
                                state,
                                Some(route_head),
                                demux_targets_ref,
                                captures_ref,
                                captured_count,
                            )
                        }
                    })
                    .collect();
                let sub_refs = branch_refs(&sub_closures);
                let nested = self.rb.gamma_n(predicate, state, &live_ins, &sub_refs)?;
                let values: Vec<ValueId> = (0..(1 + captured_count) as u16)
                    .map(|i| nested.result(i))
                    .collect();
                Ok(BranchResult {
                    state: nested.state,
                    values,
                })
            }
        }
    }

    /// Build a head-demux leaf: the index `p` of the reached demux target,
    /// followed by, per target, that target's phi values (resolved from
    /// `exit_pred`) -- poison for targets not reached, all poison when `reached`
    /// is `None` (the arm diverged).
    fn demux_capture_leaf(
        &mut self,
        reached: Option<BasicBlockId>,
        exit_pred: Option<BasicBlockId>,
        demux_targets: &[BasicBlockId],
        captures: &[TargetCapture],
    ) -> color_eyre::Result<Vec<ValueId>> {
        let mut values = Vec::new();
        match reached {
            None => {
                values.push(self.rb.constant(I32, ConstValue::Poison));
                for capture in captures {
                    for &ty in &capture.types {
                        values.push(self.rb.constant(ty, ConstValue::Poison));
                    }
                }
            }
            Some(reached) => {
                let reached_index = demux_targets
                    .iter()
                    .position(|&target| target == reached)
                    .ok_or_else(|| {
                        color_eyre::eyre::eyre!(
                            "demux arm reached {} which is not a demux target",
                            reached.0
                        )
                    })?;
                values.push(self.rb.constant(I32, ConstValue::Int(reached_index as i64)));
                for (index, capture) in captures.iter().enumerate() {
                    if index == reached_index {
                        values.extend(self.resolve_arm_join_phis(&capture.phis, exit_pred)?);
                    } else {
                        for &ty in &capture.types {
                            values.push(self.rb.constant(ty, ConstValue::Poison));
                        }
                    }
                }
            }
        }
        Ok(values)
    }

    /// Build the control predicate for the branch terminating `head`: a 2-way
    /// `bool` predicate for `CondBr`, the switch match predicate for `Switch`.
    pub(in crate::llvm_parser) fn branch_predicate(
        &mut self,
        head: BasicBlockId,
    ) -> color_eyre::Result<ValueId> {
        let bb = &self.fn_ctx.func.basic_blocks[head.0 as usize];
        match &bb.term {
            llvm_ir::Terminator::CondBr(cond_br) => {
                let condition = self.operand(&cond_br.condition)?;
                Ok(self.rb.bool_predicate(condition))
            }
            llvm_ir::Terminator::Switch(switch) => {
                let (predicate, _targets) = self.switch_predicate(switch)?;
                Ok(predicate)
            }
            other => Err(color_eyre::eyre::eyre!(
                "branch_predicate at block {} whose terminator is {:?}",
                head.0,
                other
            )),
        }
    }

    /// Emit a non-reconverging branch whose arms all return or diverge: a gamma
    /// merges each arm's return value(s); the function returns the gamma output.
    /// Returns the post-gamma state and the merged return values.
    pub(in crate::llvm_parser) fn construct_return_gamma(
        &mut self,
        head: BasicBlockId,
        arms: &[SeqRegion],
        state: State,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<(State, Vec<ValueId>)> {
        let ret_ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&self.fn_ctx.func.return_type, self.fn_ctx.llvm_mod)?;
        let arity: u16 = if ret_ty == VOID { 0 } else { 1 };
        let arm_targets = arm_target_blocks(self.fn_ctx, head)?;
        let (live_in_names, live_ins) =
            self.live_ins_for_arms(&arm_targets, boundary, &[], Some(head));
        let predicate = self.branch_predicate(head)?;

        let fn_ctx = self.fn_ctx;
        let names_ref: &[Name] = &live_in_names;
        let boundary_ref: &[BasicBlockId] = boundary;

        let arm_closures: Vec<_> = arms
            .iter()
            .map(|arm_region| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    let mut arm = RegionLowerer::arm_child(rb, fn_ctx, names_ref);
                    match arm.construct(arm_region, state, Some(head), boundary_ref)? {
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
                            "non-reconverging arm unexpectedly reached {}",
                            reached.0
                        )),
                    }
                }
            })
            .collect();
        let refs = branch_refs(&arm_closures);
        let result = self.rb.gamma_n(predicate, state, &live_ins, &refs)?;
        let values: Vec<ValueId> = (0..arity).map(|i| result.result(i)).collect();
        Ok((result.state, values))
    }
}
