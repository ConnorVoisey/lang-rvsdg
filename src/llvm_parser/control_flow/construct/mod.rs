//! **Phase 2 of the control-flow pipeline: the construction walk.** Walks a
//! Structured Region Tree ([`super::rst`]) and emits the RVSDG via the region
//! builder. The walk is mechanical -- it never inspects dominators, strongly
//! connected components, or continuation points; those decisions are already
//! encoded in the RST. Per-region live-ins are computed here (they need the
//! symbol table at emit time), reusing the shared analyses in [`super::analysis`].
//!
//! This module holds the region dispatch ([`RegionLowerer::construct`] /
//! `lower_items`) plus the small lowering primitives both halves of the walk
//! share. The two halves are split out: [`gamma`] builds branch (gamma) nodes,
//! [`theta`] builds loop (theta) nodes.

pub mod gamma;
pub mod theta;

use llvm_ir::{Instruction, Name, instruction::Phi};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::BasicBlockId,
        control_flow::{
            analysis::signature::{
                collect_walked_blocks, phi_incoming_from, phi_instructions_at, region_live_ins,
            },
            rst::{RegionItem, SeqExit, SeqRegion},
        },
        instructions::RegionLowerer,
    },
    rvsdg::{
        State, ValueId,
        builder::{BranchResult, RegionBuilder},
        types::TypeRef,
    },
};

/// Seed an arm's `name -> param(base + i)` map for `names`, in order. Every gamma/theta
/// arm seeds its region params from one or more such name lists; this is the
/// shared body of that loop.
pub(in crate::llvm_parser) fn seed_params(
    rb: &RegionBuilder,
    names: &[Name],
    base: u32,
    name_to_value: &mut FxHashMap<Name, ValueId>,
) {
    for (i, name) in names.iter().enumerate() {
        name_to_value.insert(name.clone(), rb.param(base + i as u32));
    }
}

/// The body of one gamma alternative: given the arm's region builder, lower that
/// alternative and return its resulting state and output values. Every
/// gamma-emitting site builds a list of these (as concrete closures) and hands
/// their references to `gamma_n` via [`branch_refs`].
pub(in crate::llvm_parser) type ArmBody<'a> =
    dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult> + 'a;

/// Coerce a slice of owned arm closures into the `&dyn` [`ArmBody`] references
/// `gamma_n` wants. Shared by every gamma-emitting site.
pub(in crate::llvm_parser) fn branch_refs<'a, F>(closures: &'a [F]) -> Vec<&'a ArmBody<'a>>
where
    F: Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>,
{
    closures.iter().map(|closure| closure as &ArmBody).collect()
}

/// One demux target's capture layout. The head gamma emits, per target, its
/// captured values; the demux gamma reads them back. `phis` are the phis carried
/// (empty for a loop-boundary target, whose whole leaf is captured instead);
/// `types` are the captured slots' RVSDG types (the phi types, or the full leaf
/// types for a boundary); `offset` is the index of this target's first captured
/// slot in the flattened capture vector.
pub(in crate::llvm_parser) struct TargetCapture<'m> {
    pub phis: SmallVec<[&'m Phi; 4]>,
    pub types: Vec<TypeRef>,
    pub offset: usize,
}

/// The result of constructing one region: how control left it.
pub(in crate::llvm_parser) enum ConstructExit {
    /// The region returned from the function with these operand values.
    Returned { state: State, values: Vec<ValueId> },
    /// The region reached an enclosing continuation/boundary block `reached`,
    /// arriving from `exit_pred` (`None` when its phis are already bound).
    AtBoundary {
        state: State,
        reached: BasicBlockId,
        exit_pred: Option<BasicBlockId>,
    },
    /// The region diverged before reaching any continuation.
    Diverge { state: State },
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Build a child-region lowerer over `rb` whose symbol table is seeded from
    /// `names` at param offset 0 -- the common gamma/theta arm setup. Arms that
    /// also bind captured phis or seed at a further offset build the map inline.
    pub(in crate::llvm_parser) fn arm_child<'c>(
        rb: &'c mut RegionBuilder<'g>,
        fn_ctx: &'m FnCtx<'m>,
        names: &[Name],
    ) -> RegionLowerer<'c, 'g, 'm> {
        let mut ntv = FxHashMap::default();
        seed_params(rb, names, 0, &mut ntv);
        RegionLowerer::new_child(rb, fn_ctx, ntv)
    }

    /// Construct `region`, threading `entry_state`. `entry_prev` is the block
    /// control arrived from (for binding the first block's phis); `boundary` is
    /// the enclosing region's continuation/exit blocks (used for live-in scans of
    /// nested constructs).
    pub(in crate::llvm_parser) fn construct(
        &mut self,
        region: &SeqRegion,
        entry_state: State,
        entry_prev: Option<BasicBlockId>,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<ConstructExit> {
        let (state, _prev) = self.lower_items(&region.items, entry_state, entry_prev, boundary)?;

        match &region.exit {
            SeqExit::Return { block } => {
                let bb = &self.fn_ctx.func.basic_blocks[block.0 as usize];
                let values = match &bb.term {
                    llvm_ir::Terminator::Ret(ret) => match &ret.return_operand {
                        Some(operand) => vec![self.operand(operand)?],
                        None => Vec::new(),
                    },
                    other => {
                        return Err(color_eyre::eyre::eyre!(
                            "SeqExit::Return at block {} whose terminator is {:?}",
                            block.0,
                            other
                        ));
                    }
                };
                Ok(ConstructExit::Returned { state, values })
            }
            SeqExit::ToContinuation { reached, via } => Ok(ConstructExit::AtBoundary {
                state,
                reached: *reached,
                exit_pred: *via,
            }),
            SeqExit::Diverge => Ok(ConstructExit::Diverge { state }),
            SeqExit::ReturnGamma { head, arms } => {
                let (state, values) = self.construct_return_gamma(*head, arms, state, boundary)?;
                Ok(ConstructExit::Returned { state, values })
            }
        }
    }

    /// Lower a region's straight-line/gamma/theta `items`, threading state.
    /// Returns the post-items state and the last block lowered (the linear
    /// predecessor for the region's exit). The region's `exit` is handled by the
    /// caller (the item list is role-agnostic, so this is shared by every role).
    pub(in crate::llvm_parser) fn lower_items(
        &mut self,
        items: &[RegionItem],
        entry_state: State,
        entry_prev: Option<BasicBlockId>,
        boundary: &[BasicBlockId],
    ) -> color_eyre::Result<(State, Option<BasicBlockId>)> {
        let mut state = entry_state;
        let mut prev = entry_prev;
        for item in items {
            match item {
                RegionItem::Block(block) => {
                    if let Some(pred) = prev {
                        self.bind_phis_from_pred(*block, pred)?;
                    }
                    state = self.lower_block_instructions(state, *block)?;
                    prev = Some(*block);
                }
                RegionItem::Gamma(gamma) => {
                    state = self.construct_gamma(gamma, state, boundary)?;
                    prev = None;
                }
                RegionItem::Theta(theta) => {
                    state = self.construct_theta(theta, state, boundary)?;
                    prev = None;
                }
            }
        }
        Ok((state, prev))
    }

    /// Lower every non-phi instruction of `block` in order, threading state. phis
    /// are merge nodes consumed at region boundaries (the phi-driven signature),
    /// never lowered as operations here.
    pub(in crate::llvm_parser) fn lower_block_instructions(
        &mut self,
        mut state: State,
        block: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let bb = &self.fn_ctx.func.basic_blocks[block.0 as usize];
        for inst in &bb.instrs {
            if matches!(inst, Instruction::Phi(_)) {
                continue;
            }
            state = self.lower_instruction(state, inst)?;
        }
        Ok(state)
    }

    /// Bind `block`'s phi destinations to their incoming value along the edge from
    /// `pred` -- the path-aware interior-join resolution. A phi with no incoming
    /// for `pred` is left unbound. Shared by the region/loop/entry walkers.
    pub(in crate::llvm_parser) fn bind_phis_from_pred(
        &mut self,
        block: BasicBlockId,
        pred: BasicBlockId,
    ) -> color_eyre::Result<()> {
        let bb = &self.fn_ctx.func.basic_blocks[block.0 as usize];
        for phi in &phi_instructions_at(bb) {
            if let Some((operand, _)) =
                phi_incoming_from(phi, self.fn_ctx.bb_mapper, |block| block == pred)
            {
                let value = self.operand(operand)?;
                self.name_to_value.insert(phi.dest.clone(), value);
            }
        }
        Ok(())
    }

    /// The RVSDG types of `phis`, in order -- used to poison-fill the slots an
    /// arm did not reach when emitting a gamma.
    pub(in crate::llvm_parser) fn convert_phi_types(
        &mut self,
        phis: &[&Phi],
    ) -> color_eyre::Result<Vec<TypeRef>> {
        let mut phi_types = Vec::with_capacity(phis.len());
        for phi in phis {
            phi_types.push(
                self.rb
                    .graph
                    .types
                    .convert_type_ref(&phi.to_type, self.fn_ctx.llvm_mod)?,
            );
        }
        Ok(phi_types)
    }

    /// The live-ins of the region a set of branch arms walk: the union of each
    /// arm's walked block set (bounded by `boundary`), scanned for values used
    /// inside but defined outside (`region_live_ins`, also seeding `phis`'
    /// incomings and the `pass_through_pred` edge). Returns parallel name/value
    /// vectors for seeding each arm's params.
    pub(in crate::llvm_parser) fn live_ins_for_arms(
        &self,
        arm_targets: &[BasicBlockId],
        boundary: &[BasicBlockId],
        phis: &[&Phi],
        pass_through_pred: Option<BasicBlockId>,
    ) -> (Vec<Name>, Vec<ValueId>) {
        let walked: FxHashSet<BasicBlockId> = arm_targets
            .iter()
            .flat_map(|&target| collect_walked_blocks(self.fn_ctx, target, boundary))
            .collect();
        let walked_vec: Vec<BasicBlockId> = walked.into_iter().collect();
        region_live_ins(
            self.fn_ctx,
            &self.name_to_value,
            &walked_vec,
            phis,
            pass_through_pred,
        )
    }
}
