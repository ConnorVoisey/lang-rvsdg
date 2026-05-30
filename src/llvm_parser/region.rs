use llvm_ir::{
    Instruction, Name, Operand,
    instruction::Phi,
    terminator::{CondBr, Switch},
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::BasicBlockId,
        dominance::dominates,
        instructions::{RegionLowerer, for_each_operand, instruction_dest},
        loop_arcs::LoopArcs,
        strongly_connected_components::SccId,
    },
    rvsdg::{
        ICmpPred, State, ValueId,
        builder::{BranchResult, RegionBuilder},
        types::I32,
    },
};

/// What a `lower_region` call produced at its exit point.
///
/// - `AtBoundary` is returned when the region exits at its `end` block (a
///   gamma-arm join) or at the synthetic function exit. There are no result
///   values to wire out: a gamma arm computes its result values from phi
///   contributions in the join block, not from a terminator.
/// - `Returned` is returned when the region terminated via `Ret`, carrying
///   the function's return operand (empty for void returns).
#[derive(Debug)]
pub enum RegionExit {
    AtBoundary(State),
    Returned { state: State, values: Vec<ValueId> },
}

/// A single loop in the function's CFG: the SCC that forms it, the block we
/// enter through, the block we exit to, and the classification of its arcs
/// (natural vs irregular). Built by `RegionLowerer::loop_at` when a block
/// turns out to be the header of a non-trivial SCC.
#[derive(Debug)]
struct LoopUnit {
    scc_id: SccId,
    entry: BasicBlockId,
    exit_target: BasicBlockId,
    loop_arcs: LoopArcs,
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Lower a region from `start` to `end` (exclusive) or to a function
    /// exit (`Ret` / `Unreachable` / the synthetic exit block), threading
    /// state through each instruction and each nested gamma/theta node.
    ///
    /// The returned `RegionExit` tells the caller which kind of exit
    /// happened. The caller wires this into the enclosing region: a
    /// gamma-arm `BranchResult`, a theta result, or a function `FnResult`.
    pub fn lower_region(
        &mut self,
        entry_state: State,
        start: BasicBlockId,
        end: Option<BasicBlockId>,
    ) -> color_eyre::Result<RegionExit> {
        let mut current = start;
        let mut state = entry_state;

        loop {
            // Hit the region's caller-supplied boundary (typically a
            // gamma-arm join block).
            if end == Some(current) {
                return Ok(RegionExit::AtBoundary(state));
            }

            // Hit the synthetic function-exit block (added by the parser to
            // give every `Ret` a common destination).
            if current == self.fn_ctx.exit_block_id {
                return Ok(RegionExit::AtBoundary(state));
            }

            // A block that's the header of a non-trivial SCC starts a loop;
            // lower the whole SCC as one theta node and resume after it.
            if let Some(loop_unit) = self.loop_at(current) {
                state = self.lower_loop(state, &loop_unit)?;
                current = loop_unit.exit_target;
                continue;
            }

            // Straight-line block: lower its non-phi instructions, then
            // dispatch on the terminator. Phis are absorbed into gamma-arm
            // result wiring elsewhere (see `arm_phi_contributions`).
            let bb = &self.fn_ctx.func.basic_blocks[current.0 as usize];
            for inst in &bb.instrs {
                if matches!(inst, Instruction::Phi(_)) {
                    continue;
                }
                state = self.lower_instruction(state, inst)?;
            }

            match &bb.term {
                llvm_ir::Terminator::Ret(ret) => {
                    let values = match &ret.return_operand {
                        Some(op) => vec![self.operand(op)?],
                        None => Vec::new(),
                    };
                    return Ok(RegionExit::Returned { state, values });
                }
                llvm_ir::Terminator::Br(br) => {
                    let next = self.fn_ctx.bb_mapper.get_expect(&br.dest);
                    current = *next;
                }
                llvm_ir::Terminator::CondBr(cond_br) => {
                    // Two-way conditional branch lowers to a binary gamma.
                    let join = self.fn_ctx.post_immediate_dominators[current.0 as usize]
                        .expect("branch must have a post dominator if it has a condition br");
                    state = self.lower_cond_branch(state, cond_br, join)?;
                    current = join;
                }
                llvm_ir::Terminator::Switch(switch) => {
                    // N-way branch lowers to an n-arm gamma.
                    let join = self.fn_ctx.post_immediate_dominators[current.0 as usize]
                        .expect("switch must have a post dominator if it has a condition br");
                    state = self.lower_switch(state, switch, join)?;
                    current = join;
                }
                llvm_ir::Terminator::Unreachable(_) => {
                    return Ok(RegionExit::AtBoundary(state));
                }
                t => todo!("handle terminator: {t:?}"),
            }
        }
    }

    /// If `id` is the entry of a non-trivial SCC, return the `LoopUnit`
    /// describing that loop. Trivial SCCs (single block, no back-edge) are
    /// not loops, so the function returns `None`. Returns `None` for blocks
    /// inside a loop but not at the entry.
    fn loop_at(&self, id: BasicBlockId) -> Option<LoopUnit> {
        let scc_id = self.fn_ctx.scc_analysis.get_scc_from_block(id);
        let scc = self.fn_ctx.scc_analysis.get_scc(scc_id);

        // non trivial scc's aren't loops
        if scc.is_trivial {
            return None;
        }
        let loop_arcs = LoopArcs::from_scc(scc, self.fn_ctx.bb_mapper);
        let entry = match &loop_arcs {
            LoopArcs::Natural { header, .. } => *header,
            LoopArcs::Irregular { entry_blocks, .. } => entry_blocks[0],
        };
        // Inside a loop but not at the entry: not a loop-header.
        if entry != id {
            return None;
        }

        let exit_target = match &loop_arcs {
            LoopArcs::Natural { exit, .. } => *exit,
            LoopArcs::Irregular { exit_blocks, .. } => exit_blocks[0],
        };
        Some(LoopUnit {
            scc_id,
            entry,
            exit_target,
            loop_arcs,
        })
    }

    /// Lower a loop SCC into a theta node. Not yet implemented.
    fn lower_loop(&self, _: State, _: &LoopUnit) -> color_eyre::Result<State> {
        todo!()
    }

    /// Lower a two-way conditional branch by delegating to the shared
    /// n-way gamma lowering with two arms.
    fn lower_cond_branch(
        &mut self,
        state: State,
        cond_branch: &CondBr,
        join: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let predicate = self.operand(&cond_branch.condition)?;
        let true_target = *self.fn_ctx.bb_mapper.get_expect(&cond_branch.true_dest);
        let false_target = *self.fn_ctx.bb_mapper.get_expect(&cond_branch.false_dest);
        self.lower_n_way_branch(state, predicate, &[true_target, false_target], join)
    }

    /// Lower an n-way switch by building a gamma whose arm 0 is the
    /// default destination and arms `1..=N` are the case destinations.
    ///
    /// LLVM switches match arbitrary case values, but the gamma codegen
    /// (see `lower_to_llvm/gamma.rs`) emits an LLVM `switch cond, default
    /// [(1, b1), (2, b2), ...]` where arm `i` is selected when the
    /// condition equals `i`. To bridge that, we compute an "arm index" by
    /// chaining `icmp eq` + `select` over the case-value list: the index
    /// is `i+1` if the switch operand equals the i-th case value, else 0
    /// (the default).
    fn lower_switch(
        &mut self,
        state: State,
        switch: &Switch,
        join: BasicBlockId,
    ) -> color_eyre::Result<State> {
        let switch_operand = self.operand(&switch.operand)?;

        // Arm 0 is the default; arms 1..=N are case destinations in
        // declaration order.
        let mut targets = Vec::with_capacity(switch.dests.len() + 1);
        targets.push(*self.fn_ctx.bb_mapper.get_expect(&switch.default_dest));
        for (_, dest_name) in &switch.dests {
            targets.push(*self.fn_ctx.bb_mapper.get_expect(dest_name));
        }

        // Compute the arm-index selector. Start at 0 (default) and for
        // each case `i`, replace with `i+1` when the switch operand
        // matches that case's value.
        let mut selector = self.rb.const_i32(0);
        for (i, (case_const, _)) in switch.dests.iter().enumerate() {
            let case_value = self.operand(&Operand::ConstantOperand(case_const.clone()))?;
            let matched = self.rb.icmp(ICmpPred::Eq, switch_operand, case_value);
            let case_index = self.rb.const_i32((i + 1) as i32);
            selector = self.rb.ternary(matched, case_index, selector, I32);
        }

        self.lower_n_way_branch(state, selector, &targets, join)
    }

    /// Shared body of `lower_cond_branch` and `lower_switch`: emit an
    /// n-arm gamma node and wire the join block's phis into the outer
    /// scope.
    ///
    /// For each arm target:
    ///   1. Compute the set of blocks the arm owns (dominated by the
    ///      target, not by the join).
    ///   2. Scan all arms together for SSA values used inside but defined
    ///      outside (the gamma's live-in inputs).
    ///   3. Build one closure per arm; each closure lowers its arm region
    ///      via `lower_arm`, which also resolves its contribution to each
    ///      join phi.
    ///   4. Emit the gamma via `gamma_n`, then bind each join-phi's
    ///      destination name to the corresponding gamma output.
    fn lower_n_way_branch(
        &mut self,
        state: State,
        predicate: ValueId,
        arm_targets: &[BasicBlockId],
        join: BasicBlockId,
    ) -> color_eyre::Result<State> {
        // The join block's phis define the gamma's per-arm result shape.
        let phis_at_join = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);

        // Per-arm block sets, owned so each closure can borrow into its arm.
        let arm_blocks_per_arm: Vec<FxHashSet<BasicBlockId>> = arm_targets
            .iter()
            .map(|&target| self.collect_arm_blocks(target, join))
            .collect();

        // Live-in scan needs every block across all arms. Flatten into a
        // SmallVec; arms are typically 1-3 blocks each.
        let combined_arm_blocks: SmallVec<[BasicBlockId; 8]> = arm_blocks_per_arm
            .iter()
            .flat_map(|set| set.iter().copied())
            .collect();
        let (live_in_names, live_ins) =
            self.compute_arm_live_ins(&combined_arm_blocks, &phis_at_join);

        // Pre-bind into locals so closures don't capture `self`.
        let fn_ctx = self.fn_ctx;
        let phis_slice: &[&Phi] = &phis_at_join;
        let live_in_names_slice: &[Name] = &live_in_names;

        // Owned per-arm closures. Same closure type per iteration so they
        // can live together in a `Vec<_>`. Each borrows its arm-block set
        // from `arm_blocks_per_arm`, which stays alive in this stack
        // frame for the duration of the gamma_n call.
        let arm_closures: Vec<_> = arm_targets
            .iter()
            .zip(arm_blocks_per_arm.iter())
            .map(|(&target, arm_blocks)| {
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    lower_arm(
                        rb,
                        fn_ctx,
                        state,
                        phis_slice,
                        target,
                        join,
                        arm_blocks,
                        live_in_names_slice,
                    )
                }
            })
            .collect();

        // Coerce each closure to a trait-object reference for gamma_n.
        let branch_refs: Vec<&dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>> =
            arm_closures
                .iter()
                .map(|c| c as &dyn Fn(&mut RegionBuilder) -> color_eyre::Result<BranchResult>)
                .collect();
        let result = self.rb.gamma_n(predicate, state, &live_ins, &branch_refs)?;

        // Bind each join-phi's destination to the corresponding gamma
        // output so downstream uses resolve correctly.
        for (i, phi) in phis_at_join.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), result.result(i as u16));
        }

        Ok(result.state)
    }
    /// Scan a set of arm blocks to find SSA values that are USED inside but
    /// DEFINED outside. These become the gamma node's inputs (live-ins).
    ///
    /// Returns parallel vectors so the caller can correlate names with their
    /// outer `ValueId`s: `live_in_names[i]` is the LLVM Name; `live_ins[i]`
    /// is its resolution in the outer scope, both at the same index. The
    /// closures that build each arm's region will seed their `name_to_value`
    /// with `name -> arm_rb.param(i)`, mapping the outer `ValueId` to a
    /// region parameter.
    ///
    /// Two passes for clarity over fewer allocations:
    ///   1. Walk every instruction's *dest* to build `defined_inside`.
    ///   2. Walk every instruction's *operands* via `for_each_operand`; any
    ///      `LocalOperand` whose name isn't in `defined_inside` is a
    ///      candidate live-in. Resolve via `self.name_to_value`; skip if
    ///      unknown there (the caller will hit a clearer error during arm
    ///      lowering rather than us silently producing a wrong gamma input).
    ///
    /// Phi operands at the join also need to be considered: each arm
    /// contributes one operand per phi, and that operand is resolved inside
    /// the arm's region (so its name must be in the arm's `name_to_value`).
    /// We scan phi operands whose predecessor block sits in *any* arm-block
    /// set; both arms share the unified live-in list so it's safe to merge.
    ///
    /// `defined_inside` and `seen` hold `&Name` references into the
    /// function's basic blocks, which outlive this call. No Name cloning
    /// happens until an entry is actually pushed onto the output Vec.
    ///
    /// For a switch with N arms, the caller flattens all arm-block sets
    /// into a single slice before calling; allocation cost is one short Vec
    /// per gamma.
    fn compute_arm_live_ins(
        &self,
        arm_block_set: &[BasicBlockId],
        phis_at_join: &[&Phi],
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

        // Phi operands at the join: each arm contributes one per phi. The
        // arm's closure resolves that operand inside the arm region, so the
        // name must be in the arm's `name_to_value`, which we seed from
        // live-ins. Membership is a linear scan on `arm_block_set`; arms are
        // typically 1-8 blocks, so the slice scan beats allocating a hash
        // set just for this check.
        for phi in phis_at_join {
            for (op, pred_name) in &phi.incoming_values {
                let Some(&pred_id) = self.fn_ctx.bb_mapper.get(pred_name) else {
                    continue;
                };
                if !arm_block_set.contains(&pred_id) {
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
    /// Blocks belonging to one gamma arm: those dominated by `arm_root` but
    /// not by `join`. The result is the set of blocks the arm's region will
    /// own.
    fn collect_arm_blocks(
        &self,
        arm_root: BasicBlockId,
        join: BasicBlockId,
    ) -> FxHashSet<BasicBlockId> {
        let mut res = FxHashSet::default();
        for i in 0..self.fn_ctx.func.basic_blocks.len() {
            let bb_id = BasicBlockId(i as u32);
            if dominates(arm_root, bb_id, self.fn_ctx.immediate_dominators)
                && !dominates(join, bb_id, self.fn_ctx.immediate_dominators)
            {
                res.insert(bb_id);
            }
        }
        res
    }

    /// For each phi at the join, find the operand contributed by *this*
    /// arm and resolve it in the arm's region scope.
    ///
    /// A phi's `incoming_values` is a `Vec<(Operand, Name)>` where the
    /// `Name` is the predecessor block. We pick the entry whose predecessor
    /// block lies in `arm_blocks` (looked up via the function's
    /// `BasicBlockMapper`).
    ///
    /// Returns a `Vec<ValueId>` aligned with `phis_at_join`: index `i` is
    /// this arm's contribution to the i-th phi. The caller binds these as
    /// the gamma node's arm results.
    ///
    /// Errors if any phi has no incoming pair whose predecessor sits in
    /// this arm. That indicates malformed IR or a stale `arm_blocks`;
    /// surface it loudly rather than producing an incomplete RVSDG.
    fn arm_phi_contributions(
        &mut self,
        phis_at_join: &[&Phi],
        arm_blocks: &FxHashSet<BasicBlockId>,
    ) -> color_eyre::Result<Vec<ValueId>> {
        phis_at_join
            .iter()
            .map(|phi| {
                let (operand, _pred) = phi
                    .incoming_values
                    .iter()
                    .find(|(_, pred_name)| {
                        // Phi predecessors are stored as Names. Translate to
                        // BasicBlockIds via the mapper, then check membership.
                        match self.fn_ctx.bb_mapper.get(pred_name) {
                            Some(&pred_id) => arm_blocks.contains(&pred_id),
                            None => false,
                        }
                    })
                    .ok_or_else(|| {
                        color_eyre::eyre::eyre!(
                            "phi {:?} at join has no incoming value from this arm \
                         (predecessors in phi: {:?})",
                            phi.dest,
                            phi.incoming_values
                                .iter()
                                .map(|(_, p)| p)
                                .collect::<Vec<_>>(),
                        )
                    })?;
                self.operand(operand)
            })
            .collect()
    }
}

/// Lower a single gamma arm (or, eventually, a theta sub-region) into the
/// `BranchResult` shape `gamma`/`gamma_n` expects.
///
/// Builds a fresh `RegionLowerer` whose `name_to_value` is seeded with the
/// arm's region parameters (one per live-in, in `live_in_names` order),
/// lowers the arm's blocks up to but not including `join`, then resolves
/// each join-phi's contribution from this arm via `arm_phi_contributions`.
///
/// A `Returned` exit from the arm region means an arm hit `Ret` before
/// reaching the join. That early-return shape isn't supported yet; we
/// surface it as an error rather than silently dropping return values.
fn lower_arm(
    rb: &mut RegionBuilder,
    fn_ctx: &FnCtx,
    state: State,
    phis_at_join: &[&Phi],
    arm_root: BasicBlockId,
    join: BasicBlockId,
    arm_blocks: &FxHashSet<BasicBlockId>,
    live_in_names: &[Name],
) -> color_eyre::Result<BranchResult> {
    let mut name_to_value = FxHashMap::default();
    for (i, name) in live_in_names.iter().enumerate() {
        name_to_value.insert(name.clone(), rb.param(i as u32));
    }

    let mut arm = RegionLowerer {
        rb,
        fn_ctx,
        name_to_value,
    };

    let state = match arm.lower_region(state, arm_root, Some(join))? {
        RegionExit::AtBoundary(state) => state,
        RegionExit::Returned { .. } => {
            return Err(color_eyre::eyre::eyre!(
                "early returns are not supported within gamma-arm lowering"
            ));
        }
    };

    let values = arm.arm_phi_contributions(phis_at_join, arm_blocks)?;

    Ok(BranchResult { state, values })
}

/// Return the leading run of `Phi` instructions at the start of `basic_block`.
/// LLVM IR requires all phis to appear contiguously at the start of a
/// block; we stop at the first non-phi.
fn phi_instructions_at(
    basic_block: &llvm_ir::BasicBlock,
) -> SmallVec<[&llvm_ir::instruction::Phi; 4]> {
    basic_block
        .instrs
        .iter()
        .map_while(|i| match i {
            Instruction::Phi(p) => Some(p),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        llvm_parser::{
            block_mapper::BasicBlockMapper,
            dominance::{ForwardView, ReverseView, compute_dominance},
            strongly_connected_components::SccAnalysis,
        },
        rvsdg::{Linkage, RVSDGMod, builder::RegionBuilder},
    };
    use llvm_ir::Module;
    use pretty_assertions::assert_eq;
    use std::sync::Mutex;

    // The llvm-ir parser's first call lazily initialises an attribute-kind
    // table that asserts `nocapture` is present. LLVM 19 removed `nocapture`,
    // so that init panics if it races against itself. Serialise parse calls
    // until we move off llvm-ir 0.11.3.
    static LLVM_PARSE_LOCK: Mutex<()> = Mutex::new(());

    /// Owns everything a FnCtx borrows from. Tests build one of these and
    /// either borrow a FnCtx directly (for free-function helpers like
    /// `phi_instructions_at`) or call `with_lowerer` to spin up a real
    /// `RegionLowerer` and run assertions against its methods.
    struct TestFn {
        module: Module,
        bb_mapper: BasicBlockMapper,
        scc_analysis: SccAnalysis,
        immediate_dominators: Vec<Option<BasicBlockId>>,
        post_immediate_dominators: Vec<Option<BasicBlockId>>,
        exit_block_id: BasicBlockId,
    }

    impl TestFn {
        fn from_ir(ir: &str) -> Self {
            let module = {
                let _guard = LLVM_PARSE_LOCK.lock().unwrap();
                Module::from_ir_str(ir).expect("parse test IR")
            };
            assert!(
                !module.functions.is_empty(),
                "test IR must define at least one function"
            );
            let func = &module.functions[0];
            let mut bb_mapper = BasicBlockMapper::new(func.basic_blocks.len());
            for block in &func.basic_blocks {
                bb_mapper.intern(&block.name);
            }
            let exit_name = bb_mapper.exit_name();
            let exit_block_id = bb_mapper.intern(&exit_name);
            for (i, block) in func.basic_blocks.iter().enumerate() {
                let from = BasicBlockId(i as u32);
                match &block.term {
                    llvm_ir::Terminator::Br(br) => {
                        let to = *bb_mapper.get_expect(&br.dest);
                        bb_mapper.add_connection(from, to);
                    }
                    llvm_ir::Terminator::CondBr(cb) => {
                        let t = *bb_mapper.get_expect(&cb.true_dest);
                        let f = *bb_mapper.get_expect(&cb.false_dest);
                        bb_mapper.add_connection(from, t);
                        bb_mapper.add_connection(from, f);
                    }
                    llvm_ir::Terminator::Ret(_) => {
                        bb_mapper.add_connection(from, exit_block_id);
                    }
                    llvm_ir::Terminator::Switch(sw) => {
                        let d = *bb_mapper.get_expect(&sw.default_dest);
                        bb_mapper.add_connection(from, d);
                        for (_, dest) in &sw.dests {
                            let did = *bb_mapper.get_expect(dest);
                            bb_mapper.add_connection(from, did);
                        }
                    }
                    _ => {}
                }
            }
            let scc_analysis = bb_mapper.get_strongly_connected_components();
            let immediate_dominators = compute_dominance(&ForwardView {
                nodes: &bb_mapper.blocks,
                entry: BasicBlockId(0),
            });
            let post_immediate_dominators = compute_dominance(&ReverseView {
                nodes: &bb_mapper.blocks,
                exit: exit_block_id,
            });
            Self {
                module,
                bb_mapper,
                scc_analysis,
                immediate_dominators,
                post_immediate_dominators,
                exit_block_id,
            }
        }

        fn fn_ctx(&self) -> FnCtx<'_> {
            FnCtx {
                llvm_mod: &self.module,
                func: &self.module.functions[0],
                bb_mapper: &self.bb_mapper,
                scc_analysis: &self.scc_analysis,
                immediate_dominators: &self.immediate_dominators,
                post_immediate_dominators: &self.post_immediate_dominators,
                exit_block_id: self.exit_block_id,
            }
        }

        fn block(&self, name: &str) -> BasicBlockId {
            *self
                .bb_mapper
                .get(&local_name(name))
                .unwrap_or_else(|| panic!("test IR has no block named %{name}"))
        }

        /// Spin up an `RVSDGMod` + `RegionBuilder` + `RegionLowerer` and run
        /// `f` against the lowerer. `name_to_value` seeds the lowerer's outer
        /// SSA map (stand-in for values defined before the gamma node we'd be
        /// lowering). The closure runs the assertions; nothing about the
        /// RVSDG it builds is observed, so we don't bother finishing the
        /// function definition.
        fn with_lowerer<R>(
            &self,
            name_to_value: FxHashMap<Name, ValueId>,
            f: impl FnOnce(&mut RegionLowerer<'_, '_, '_>) -> R,
        ) -> R {
            let mut rvsdg = RVSDGMod::new_host("test".to_string());
            let fn_id = rvsdg.declare_fn("test".to_string(), &[], &[], Linkage::Internal);
            let mut rb = RegionBuilder::new_from_func(&mut rvsdg, fn_id);
            let fn_ctx = self.fn_ctx();
            let mut lowerer = RegionLowerer {
                rb: &mut rb,
                name_to_value,
                fn_ctx: &fn_ctx,
            };
            f(&mut lowerer)
        }
    }

    fn local_name(s: &str) -> Name {
        Name::Name(Box::new(s.to_string()))
    }

    // ------------------------------------------------------------------------
    // phi_instructions_at
    // ------------------------------------------------------------------------

    #[test]
    fn phi_instructions_at_returns_phis_at_block_start() {
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
  %r1 = phi i32 [ %a, %t ], [ %b, %f ]
  %r2 = phi i32 [ %b, %t ], [ %a, %f ]
  %r3 = add i32 %r1, %r2
  ret i32 %r3
}
"#,
        );
        let j = test_fn.block("j");
        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        assert_eq!(phis.len(), 2);
        assert_eq!(phis[0].dest, local_name("r1"));
        assert_eq!(phis[1].dest, local_name("r2"));
    }

    #[test]
    fn phi_instructions_at_empty_when_no_phi() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %a) {
entry:
  ret i32 %a
}
"#,
        );
        let entry = test_fn.block("entry");
        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[entry.0 as usize]);
        assert!(phis.is_empty());
    }

    #[test]
    fn phi_instructions_at_stops_at_first_non_phi() {
        // Only the phi prefix is taken; if a non-phi sits between phis (which
        // is invalid LLVM but we should still defend), we stop early.
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
  %r1 = phi i32 [ %a, %t ], [ %b, %f ]
  %s = add i32 %r1, 1
  ret i32 %s
}
"#,
        );
        let j = test_fn.block("j");
        let phis = phi_instructions_at(&test_fn.fn_ctx().func.basic_blocks[j.0 as usize]);
        assert_eq!(phis.len(), 1);
        assert_eq!(phis[0].dest, local_name("r1"));
    }

    // ------------------------------------------------------------------------
    // collect_arm_blocks
    // ------------------------------------------------------------------------

    #[test]
    fn collect_arm_blocks_simple_diamond() {
        // entry -> t -> j
        //       -> f -> j
        // Each arm is a single block dominated by its arm root.
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
            let true_arm = lowerer.collect_arm_blocks(t, j);
            assert_eq!(true_arm.len(), 1);
            assert!(true_arm.contains(&t));

            let false_arm = lowerer.collect_arm_blocks(f, j);
            assert_eq!(false_arm.len(), 1);
            assert!(false_arm.contains(&f));
        });
    }

    #[test]
    fn collect_arm_blocks_includes_dominated_blocks() {
        // entry -> t -> mid -> j
        //       -> f -> j
        // The true arm spans both `t` and `mid` (both dominated by t and not
        // by j).
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
            let true_arm = lowerer.collect_arm_blocks(t, j);
            assert_eq!(true_arm.len(), 2);
            assert!(true_arm.contains(&t));
            assert!(true_arm.contains(&mid));
            assert!(!true_arm.contains(&j), "join must not be in the arm");
            assert!(!true_arm.contains(&f), "other arm must not be in the arm");
        });
    }

    #[test]
    fn collect_arm_blocks_excludes_blocks_past_join() {
        // entry -> t -> j -> after
        //       -> f -> j
        // `after` is past the join; it must not appear in either arm.
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
            let true_arm = lowerer.collect_arm_blocks(t, j);
            assert!(
                !true_arm.contains(&after),
                "post-join block must not be in arm"
            );
        });
    }

    // ------------------------------------------------------------------------
    // compute_arm_live_ins
    // ------------------------------------------------------------------------

    #[test]
    fn live_ins_include_phi_operands_from_function_params() {
        // The phi at the join references function params %a and %b; both must
        // be picked up as live-ins (this is the regression that caused the
        // earlier crash on the `max` example).
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
            let (names, values) = lowerer.compute_arm_live_ins(&arm_blocks, &phis);
            assert!(names.contains(&local_name("a")), "%a should be a live-in");
            assert!(names.contains(&local_name("b")), "%b should be a live-in");
            assert_eq!(names.len(), values.len());
            assert_eq!(names.len(), 2);
        });
    }

    #[test]
    fn live_ins_skip_locally_defined_names() {
        // %local and %used_local are both defined inside the true arm; neither
        // should be flagged as a live-in even though they're used inside.
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
            let (names, _) = lowerer.compute_arm_live_ins(&arm_blocks, &phis);
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
        // Operands `1` and `2` are constants; only the local %a should be
        // picked up.
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
            let (names, values) = lowerer.compute_arm_live_ins(&arm_blocks, &phis);
            assert_eq!(names, vec![local_name("a")]);
            assert_eq!(values, vec![ValueId(100)]);
        });
    }

    #[test]
    fn live_ins_silently_skip_unresolved_names() {
        // outer_map is empty; %a is referenced but unresolvable. We must
        // not panic; we leave the missing-value error for the arm lowerer
        // to surface at the actual use site.
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
            let (names, values) = lowerer.compute_arm_live_ins(&arm_blocks, &phis);
            assert!(names.is_empty());
            assert!(values.is_empty());
        });
    }

    #[test]
    fn live_ins_pick_up_operands_from_arm_instructions() {
        // %x is used inside the arm but defined outside (we pretend it lives
        // in the outer scope via outer_map). It must surface as a live-in
        // even though no phi at the join mentions it.
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
            let (names, _) = lowerer.compute_arm_live_ins(&arm_blocks, &phis);
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
        // %a used twice in t and in the phi; should appear only once in the
        // live-in list.
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
            let (names, _) = lowerer.compute_arm_live_ins(&arm_blocks, &phis);
            assert_eq!(
                names.iter().filter(|n| **n == local_name("a")).count(),
                1,
                "%a should appear once even though referenced multiple times"
            );
        });
    }
}
