//! Lower a strongly connected component into a theta node.
//!
//! Construction follows Bahmann, Reissmann, Jahre, Meyer (2015) section
//! 4.1. The component's body is walked as a tree of gamma nodes; each
//! leaf of the tree produces a tuple `(next_vars..., q, r)` whose meaning
//! is:
//!
//!   - `r` is the repetition predicate: `1` on a path that ends at the
//!     repetition arc (back to the header), `0` on a path that ends at
//!     any exit arc.
//!   - `q` is the exit-vertex index, used only when `r = 0` for
//!     dispatching the post-theta resumption to the correct exit
//!     target. On repetition leaves it is set to `0` (irrelevant).
//!   - `next_vars` are the loop-variable values flowing into the next
//!     iteration (rep leaf) or out of the theta (exit leaf).
//!
//! The walker is recursive: an unconditional branch to a non-leaf
//! block continues the walk; a conditional branch builds a gamma whose
//! arms each recurse. Branches to outside the SCC become exit leaves;
//! branches back to the header become repetition leaves.
//!
//! Do-while and test-first loops are both special cases of this
//! walker (they collapse to a one-leaf and two-leaf gamma tree
//! respectively). Multi-exit components dispatch through a post-theta
//! gamma on `q` that routes control to the correct exit target.
//! Multi-back-edge components are rejected until a future phase adds
//! the demand-analysis that produces the dedicated latch transform.

use llvm_ir::{
    Instruction, Name, Operand, TypeRef as LLVMTypeRef, instruction::Phi, terminator::CondBr,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::BasicBlockId,
        dominance::dominates,
        instructions::{
            RegionLowerer, for_each_operand, for_each_terminator_operand, instruction_dest,
        },
        region::{
            RegionExit,
            branches::as_branch_refs,
            phi::{phi_incoming_from, phi_instructions_at},
        },
        scc_tree::SccTreeNodeId,
    },
    rvsdg::{
        ConstValue, State, ValueId,
        builder::{BranchResult, LoopResult, RegionBuilder},
        types::{BOOL, ScalarType, TypeRef},
    },
};
use color_eyre::eyre::{bail, eyre};

/// Metadata for one phi at the loop's header block. Each header phi
/// becomes a loop_var slot in the theta. The preheader operand resolves
/// to `init` (outer-scope); the in-SCC operand is `next_operand`,
/// resolved later inside the body's scope to produce the slot's
/// back-edge value.
#[derive(Debug)]
pub(super) struct HeaderPhiInfo<'m> {
    pub(super) phi: &'m Phi,
    pub(super) init: ValueId,
    pub(super) next_operand: &'m Operand,
}

/// How a loop-closed phi at an exit target should be bound after the
/// theta. Computed during `analyze_loop`; consumed in
/// `lower_scc_as_theta`'s post-theta wiring.
#[derive(Debug, Clone, Copy)]
pub(super) enum LcssaBinding {
    /// The loop-closed phi references a constant; bind directly without
    /// involving theta.
    Constant(ValueId),
    /// The loop-closed phi corresponds to header phi `index`'s slot.
    /// The theta projection at slot `index` is what the phi wants
    /// (for sub-case-A: the back-edge operand; for sub-case-B at a
    /// header-source exit: the slot's input value).
    HeaderPhi { index: u16 },
    /// The loop-closed phi references a body-internal value. An extra
    /// loop_var slot was allocated for it; resolve via
    /// `theta_result.result(N + L + index)`.
    Extra { index: u16 },
}

/// One body-internal SSA value that has to flow through the theta
/// because a loop-closed phi at some exit target references it. Each
/// instance allocates one extra loop_var slot beyond the header phi
/// and live-in slots. Stored as a single struct rather than parallel
/// vectors so the (name, type) pair can never drift apart as more
/// fields are added.
#[derive(Debug, Clone)]
pub(super) struct LcssaExtraSlot {
    pub(super) name: Name,
    #[allow(dead_code)]
    pub(super) ty: TypeRef,
}

/// One contiguous backing buffer for theta's `loop_vars`, with markers
/// partitioning it into header phis / live-ins / loop-closed extras.
/// The slice passed to `rb.theta` is `all()`; the per-category views
/// support post-theta reasoning without offset arithmetic at every
/// call site.
///
/// Single allocation; sized inline for 16 slots which covers the
/// common case (up to 8 header phis + 4 live-ins + 2 loop-closed
/// extras, with headroom).
#[derive(Debug)]
pub(super) struct LoopVarInits {
    storage: SmallVec<[ValueId; 16]>,
    header_phi_end: u16,
    live_in_end: u16,
}

impl LoopVarInits {
    pub(super) fn all(&self) -> &[ValueId] {
        &self.storage
    }
    #[allow(dead_code)]
    pub(super) fn header_phis(&self) -> &[ValueId] {
        &self.storage[..self.header_phi_end as usize]
    }
    #[allow(dead_code)]
    pub(super) fn live_ins(&self) -> &[ValueId] {
        &self.storage[self.header_phi_end as usize..self.live_in_end as usize]
    }
    #[allow(dead_code)]
    pub(super) fn lcssa_extras(&self) -> &[ValueId] {
        &self.storage[self.live_in_end as usize..]
    }
}

/// Output of `analyze_loop`: everything `lower_scc_as_theta` needs to
/// build a theta and wire its outputs.
#[derive(Debug)]
pub(super) struct LoopLowerCtx<'m> {
    /// Sum of instruction counts across the loop body. Used to pre-size
    /// the body's `name_to_value` HashMap.
    pub(super) body_instr_count: u32,

    /// The contiguous loop_vars buffer; pass `inits.all()` to
    /// `rb.theta` (along with one extra init value for the q slot).
    pub(super) inits: LoopVarInits,

    /// Header phi metadata, parallel to `inits.header_phis()`.
    pub(super) header_phis: SmallVec<[HeaderPhiInfo<'m>; 4]>,
    /// Live-in Names, parallel to `inits.live_ins()`. Used inside the
    /// theta body to seed `name_to_value`.
    pub(super) live_in_names: SmallVec<[Name; 4]>,
    /// Loop-closed-extra slots, parallel to `inits.lcssa_extras()`.
    /// Body-internal SSA values that the body must thread through theta
    /// because some exit target's loop-closed phi references them.
    pub(super) lcssa_extras: SmallVec<[LcssaExtraSlot; 2]>,
    /// Per-exit-arc loop-closed-phi bindings. Index `i` corresponds to
    /// the i-th entry in the SCC's `exit_arcs`. Each inner vector
    /// holds `(lcssa_dest_name, binding)` pairs for the loop-closed
    /// phis at that exit target whose incoming comes from the
    /// corresponding exit arc's source block.
    pub(super) lcssa_bindings_per_exit: SmallVec<[SmallVec<[(Name, LcssaBinding); 2]>; 4]>,
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Lower a strongly connected component into a theta node. Returns
    /// the state after the theta dispatch and the block to resume
    /// lowering at: either the single exit target (single-exit case)
    /// or the post-dominator join of all exit targets (multi-exit
    /// case).
    ///
    /// `entry` is the SCC entry vertex through which control arrived. For
    /// a single-entry loop it equals the sole entry block; for a
    /// multi-entry (irreducible) loop it selects which entry the theta's
    /// `q` dispatch starts at this iteration.
    pub(super) fn lower_scc_as_theta(
        &mut self,
        state: State,
        scc_id: SccTreeNodeId,
        entry: BasicBlockId,
    ) -> color_eyre::Result<(State, BasicBlockId)> {
        let arcs = &self.fn_ctx.scc_tree.arcs[scc_id.0 as usize];
        if arcs.entry_blocks.len() != 1 {
            // Irreducible / multi-entry loop: lowered via the `q`
            // entry-dispatch at the SCC's dispatch dominator
            // (`lower_multi_entry_dispatch`), reached from `lower_region`
            // before control ever arrives at an individual entry vertex.
            // Reaching it here would mean the dispatch-dominator trigger
            // was missed.
            bail!(
                "multi-entry SCC reached at entry vertex {} without going \
                 through its dispatch dominator (entries: {:?})",
                entry.0,
                arcs.entry_blocks.iter().map(|b| b.0).collect::<Vec<_>>(),
            );
        }
        if arcs.repetition_arcs.len() != 1 {
            bail!(
                "multi-back-edge strongly connected component at entry block {} \
                 ({} repetition arcs): not yet supported. Run the LLVM opt pass \
                 `loop-simplify` upstream to consolidate back-edges through a \
                 single dedicated latch, or wait for the demand-analysis pass \
                 that synthesises one inside the construction.",
                arcs.entry_blocks[0].0,
                arcs.repetition_arcs.len(),
            );
        }

        let header = arcs.entry_blocks[0];
        let exit_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]> =
            arcs.exit_arcs.iter().copied().collect();
        let n_exits = exit_arcs.len();

        // Each exit arc gets a small integer k_x. The walker emits
        // k_x into the q slot on the path that ends at that arc, and
        // the post-theta dispatcher reads q to choose which exit
        // target to resume at. Stored inline as a small linear vec
        // (a hashmap would over-allocate for the 1-4 exits a typical
        // loop has, and a linear scan is faster than a hash lookup
        // at that size).
        let exit_arc_indices: &[(BasicBlockId, BasicBlockId)] = &exit_arcs;

        // SCC body, borrowed directly from the SCC tree. The walker
        // uses `slice::contains` here rather than a hashset because
        // typical loops have well under 16 blocks in the body and a
        // linear scan over a contiguous SmallVec slice beats a hash
        // lookup at that size (and avoids the per-SCC allocation).
        let scc_body: &[BasicBlockId] = &self.fn_ctx.scc_tree.blocks[scc_id.0 as usize];

        let ctx = self.analyze_loop(scc_id, header, &exit_arcs)?;

        let n_header = ctx.header_phis.len();
        let n_live_in = ctx.live_in_names.len();
        let n_lcssa_extra = ctx.lcssa_extras.len();
        let slot_count = n_header + n_live_in + n_lcssa_extra;
        let slot_count_u16 = slot_count as u16;

        // Theta inits: the loop_vars buffer plus one extra slot for q.
        // q's init value is arbitrary: every body iteration overwrites
        // it via the walker's leaf, and only the last iteration's value
        // is read at the post-theta dispatch.
        let mut inits_with_q: SmallVec<[ValueId; 16]> = ctx.inits.all().iter().copied().collect();
        let q_init = self.rb.const_i32(0);
        inits_with_q.push(q_init);

        let fn_ctx = self.fn_ctx;
        let ctx_ref = &ctx;

        let theta_result = self.rb.theta(state, &inits_with_q, |body_rb| {
            // Seed the body's `name_to_value` with EVERY slot's name
            // (header phi dest, live-in name, loop-closed extra name)
            // mapped to the slot's input value. With every slot in
            // `name_to_value`, the body walker's leaf computation can
            // resolve slot values by name without falling back to
            // `body.rb.param(slot_idx)` (which would break inside the
            // gamma arm regions the walker builds at every CondBr).
            let cap = n_header + n_live_in + n_lcssa_extra + ctx_ref.body_instr_count as usize;
            let mut name_to_value = FxHashMap::with_capacity_and_hasher(cap, Default::default());
            for (i, info) in ctx_ref.header_phis.iter().enumerate() {
                name_to_value.insert(info.phi.dest.clone(), body_rb.param(i as u32));
            }
            for (j, name) in ctx_ref.live_in_names.iter().enumerate() {
                name_to_value.insert(name.clone(), body_rb.param((n_header + j) as u32));
            }
            for (k, slot) in ctx_ref.lcssa_extras.iter().enumerate() {
                // An extra slot that captures a header phi dest (the
                // non-header sub-case-B binding) shares that name, which is
                // already bound to its header param above. Don't overwrite
                // it: the body must keep seeing the header value, and the
                // leaf fills the extra slot from that same binding. Other
                // extras name body-internal values not yet bound, so they
                // bind normally.
                if name_to_value.contains_key(&slot.name) {
                    continue;
                }
                name_to_value.insert(
                    slot.name.clone(),
                    body_rb.param((n_header + n_live_in + k) as u32),
                );
            }

            let mut body = RegionLowerer::new_child(body_rb, fn_ctx, name_to_value);

            let walker = BodyWalker {
                header,
                scc_body,
                exit_arc_indices,
                ctx: ctx_ref,
                slot_count: slot_count_u16,
            };

            // Walk the body from the header. Returns a value tuple of
            // length `slot_count + 2`: slot values, then q, then r.
            let (final_state, values) = lower_body_walk(&mut body, state, header, None, &walker)?;

            // Theta's next_vars include the q slot but not r (which is
            // the loop's repetition predicate, returned separately).
            let predicate_r = values[slot_count + 1];
            let next_vars: Vec<ValueId> = values[..=slot_count].to_vec();

            Ok(LoopResult {
                condition: predicate_r,
                next_state: final_state,
                next_vars,
            })
        })?;

        let header_count = n_header as u16;
        let live_in_count = n_live_in as u16;

        // Bind each header phi dest to its theta projection so any
        // outer-scope code that references the loop variable's
        // post-loop value resolves correctly.
        for (i, info) in ctx.header_phis.iter().enumerate() {
            self.name_to_value
                .insert(info.phi.dest.clone(), theta_result.result(i as u16));
        }

        if n_exits == 0 {
            // Infinite loop: the theta has no exit arc, so control never
            // leaves it (the repetition predicate is always 1). Nothing
            // after the loop is reachable, so resume at the synthetic
            // function-exit block; the enclosing walk then terminates at
            // its boundary and the post-loop code is left as the dead code
            // it is (at runtime control never returns here).
            return Ok((theta_result.state, self.fn_ctx.exit_block_id));
        }

        if n_exits == 1 {
            // Single-exit shape: bind loop-closed phis directly in
            // outer scope, then resume at the single exit target.
            let (_, exit_target) = exit_arcs[0];
            let lcssas = &ctx.lcssa_bindings_per_exit[0];
            for (lcssa_dest, binding) in lcssas {
                let value = match binding {
                    LcssaBinding::Constant(v) => *v,
                    LcssaBinding::HeaderPhi { index } => theta_result.result(*index),
                    LcssaBinding::Extra { index } => {
                        theta_result.result(header_count + live_in_count + *index)
                    }
                };
                self.name_to_value.insert(lcssa_dest.clone(), value);
            }
            Ok((theta_result.state, exit_target))
        } else {
            // Multi-exit shape: dispatch on the q slot's projection to
            // route control to the right exit target. Each exit
            // target's loop-closed phis are pre-bound to their theta
            // projections inside the corresponding gamma arm.
            let q_value = theta_result.result(slot_count_u16);
            let exit_targets: SmallVec<[BasicBlockId; 4]> =
                exit_arcs.iter().map(|&(_, dst)| dst).collect();

            let join = self.compute_exit_targets_join(&exit_targets)?;

            // Materialise per-exit loop-closed bindings as resolved
            // `(name, value)` pairs so the dispatcher can hand each
            // arm its pre-binding map.
            let lcssa_per_exit_resolved: Vec<Vec<(Name, ValueId)>> = ctx
                .lcssa_bindings_per_exit
                .iter()
                .map(|exit_lcssas| {
                    exit_lcssas
                        .iter()
                        .map(|(name, binding)| {
                            let value = match binding {
                                LcssaBinding::Constant(v) => *v,
                                LcssaBinding::HeaderPhi { index } => theta_result.result(*index),
                                LcssaBinding::Extra { index } => {
                                    theta_result.result(header_count + live_in_count + *index)
                                }
                            };
                            (name.clone(), value)
                        })
                        .collect()
                })
                .collect();

            let state_after = self.lower_exit_dispatch(
                theta_result.state,
                q_value,
                &exit_targets,
                join,
                &lcssa_per_exit_resolved,
            )?;

            Ok((state_after, join))
        }
    }

    /// If `block` is the dispatch dominator of some multi-entry
    /// (irreducible) SCC — the lowest block in the dominator tree that
    /// dominates all the SCC's entry vertices — return that SCC. This is
    /// where the `q` entry-dispatch is lowered: every path from here to the
    /// loop reaches one of the entries, so `q` can be computed in the
    /// branch structure between this block and the entries.
    pub(super) fn multi_entry_dispatch_at(&self, block: BasicBlockId) -> Option<SccTreeNodeId> {
        for i in 0..self.fn_ctx.scc_tree.len() {
            let arcs = &self.fn_ctx.scc_tree.arcs[i];
            if arcs.entry_blocks.len() <= 1 {
                continue;
            }
            if self.entries_dispatch_dom(&arcs.entry_blocks) == Some(block) {
                return Some(SccTreeNodeId(i as u32));
            }
        }
        None
    }

    /// Lowest common ancestor of `entries` in the (forward) dominator tree:
    /// the deepest block that dominates every entry vertex. Walks up the
    /// first entry's immediate-dominator chain until it dominates them all.
    fn entries_dispatch_dom(&self, entries: &[BasicBlockId]) -> Option<BasicBlockId> {
        let idoms = self.fn_ctx.immediate_dominators;
        let mut cand = *entries.first()?;
        loop {
            if entries.iter().all(|&e| dominates(cand, e, idoms)) {
                return Some(cand);
            }
            match idoms[cand.0 as usize] {
                Some(parent) if parent != cand => cand = parent,
                _ => return None,
            }
        }
    }

    /// Lower an irreducible (multi-entry) strongly connected component
    /// into a theta with an auxiliary `q` entry-dispatch — Bahmann et al.
    /// 2015 §4.1's `q` predicate, built in-tree rather than via an upstream
    /// `fix-irreducible` pass.
    ///
    /// Shape: the theta carries one loop_var per phi at every entry vertex,
    /// plus a `q_entry` selector and a `q_exit` index. Each iteration the
    /// body dispatches on `q_entry` (a γ-node) to the code of one entry
    /// vertex, walks it until it reaches a repetition arc (an edge back to
    /// some entry — sets `q_entry` to that entry and repeats) or an exit
    /// arc (sets `q_exit` and stops). `entry` is the vertex control arrived
    /// through, so its phis take their real preheader value and `q_entry`
    /// starts there; the other entries' phi slots start at zero and are
    /// only read after a repetition arc has written them.
    ///
    /// This first cut bails on shapes it doesn't yet model: a nested inner
    /// SCC, a `switch`/`return` inside an entry's region, or a loop-closed
    /// value that is an entry phi (rather than a body-internal value or a
    /// constant). Those fall back to an error rather than miscompiling.
    pub(super) fn lower_multi_entry_dispatch(
        &mut self,
        state: State,
        scc_id: SccTreeNodeId,
        dispatch_dom: BasicBlockId,
    ) -> color_eyre::Result<(State, BasicBlockId)> {
        let arcs = self.fn_ctx.scc_tree.arcs[scc_id.0 as usize].clone();
        let entries: SmallVec<[BasicBlockId; 4]> = arcs.entry_blocks.clone();
        let scc_body: SmallVec<[BasicBlockId; 8]> = self.fn_ctx.scc_tree.blocks[scc_id.0 as usize]
            .iter()
            .copied()
            .collect();

        // Entry phis: every phi at every entry vertex becomes a loop_var.
        // Their init values come from the entry-region walk below.
        let mut entry_phis: SmallVec<[EntryPhi<'m>; 4]> = SmallVec::new();
        for &e in &entries {
            let bb = &self.fn_ctx.func.basic_blocks[e.0 as usize];
            for phi in phi_instructions_at(bb) {
                entry_phis.push(EntryPhi { entry: e, phi });
            }
        }
        let n_entry_phi = entry_phis.len();

        // Loop-closed values: values defined inside the SCC and used
        // outside it. Irreducible loops are left un-lcssa'd by the opt
        // pipeline, so detect direct cross-boundary uses here. Each becomes
        // an extra loop_var threaded out and bound at its outer uses. The
        // type comes straight off the using `LocalOperand`.
        let closed_typed: Vec<(Name, LLVMTypeRef)> = {
            let mut defined_inside: FxHashSet<&Name> = FxHashSet::default();
            for &b in &scc_body {
                for inst in &self.fn_ctx.func.basic_blocks[b.0 as usize].instrs {
                    if let Some(d) = instruction_dest(inst) {
                        defined_inside.insert(d);
                    }
                }
            }
            let mut out: Vec<(Name, LLVMTypeRef)> = Vec::new();
            let mut seen: FxHashSet<Name> = FxHashSet::default();
            for (i, bb) in self.fn_ctx.func.basic_blocks.iter().enumerate() {
                if scc_body.contains(&BasicBlockId(i as u32)) {
                    continue;
                }
                let mut visit = |op: &Operand| {
                    if let Operand::LocalOperand { name, ty } = op {
                        if defined_inside.contains(name) && seen.insert(name.clone()) {
                            out.push((name.clone(), ty.clone()));
                        }
                    }
                };
                for inst in &bb.instrs {
                    for_each_operand(inst, &mut visit);
                }
                for_each_terminator_operand(&bb.term, &mut visit);
            }
            out
        };
        // Extra loop_var per closed value; init is a placeholder (the slot
        // is written on the exiting iteration's leaf).
        let mut closed_extras: SmallVec<[Name; 4]> = SmallVec::new();
        let mut closed_inits: SmallVec<[ValueId; 4]> = SmallVec::new();
        for (name, llvm_ty) in &closed_typed {
            let ty = self
                .rb
                .graph
                .types
                .convert_type_ref(llvm_ty, self.fn_ctx.llvm_mod)?;
            closed_inits.push(self.zero_of(ty)?);
            closed_extras.push(name.clone());
        }
        let n_closed = closed_extras.len();
        let base = n_entry_phi + n_closed;

        // Entry-region walk from the dispatch dominator: each path ends at
        // a loop entry and yields (q, entry-phi inits); branches merge them
        // with gamma nodes. This is the paper's §4.1 `q` assignment on the
        // entry arcs, computed as a value rather than via CFG edits.
        let (state, q_and_inits) = {
            let ectx = EntryCtx {
                entries: &entries,
                entry_phis: &entry_phis,
                scc_body: &scc_body,
            };
            entry_walk(self, state, dispatch_dom, None, &ectx)?
        };
        let q_init = q_and_inits[0];

        // Theta inits: entry-phi inits (from the walk), then closed-value
        // placeholders, then q_entry (= the dispatched q) and q_exit (0).
        let mut inits: SmallVec<[ValueId; 16]> = q_and_inits[1..].iter().copied().collect();
        inits.extend(closed_inits.iter().copied());
        inits.push(q_init); // q_entry init
        inits.push(self.rb.const_i32(0)); // q_exit init

        let exit_arcs: SmallVec<[(BasicBlockId, BasicBlockId); 4]> =
            arcs.exit_arcs.iter().copied().collect();

        let fn_ctx = self.fn_ctx;
        let entries_ref = &entries;
        let scc_body_ref = &scc_body;
        let entry_phis_ref = &entry_phis;
        let closed_ref = &closed_extras;
        let exit_arcs_ref = &exit_arcs;

        let theta_result = self.rb.theta(state, &inits, |body_rb| {
            // Seed loop-var params by name. Entry phi dests, then closed
            // extras. q slots are read positionally, not by name.
            let mut name_to_value = FxHashMap::default();
            for (i, ep) in entry_phis_ref.iter().enumerate() {
                name_to_value.insert(ep.phi.dest.clone(), body_rb.param(i as u32));
            }
            for (k, name) in closed_ref.iter().enumerate() {
                name_to_value.insert(name.clone(), body_rb.param((n_entry_phi + k) as u32));
            }
            let q_entry = body_rb.param(base as u32);

            let walker = MultiWalker {
                entries: entries_ref,
                scc_body: scc_body_ref,
                entry_phis: entry_phis_ref,
                closed: closed_ref,
                exit_arcs: exit_arcs_ref,
                base,
            };

            // Dispatch on q_entry to each entry's region. Each arm walks
            // from one entry and produces the full leaf tuple
            // [base..., q_entry', q_exit, r].
            let snapshot: Vec<(Name, ValueId)> =
                name_to_value.iter().map(|(n, &v)| (n.clone(), v)).collect();
            let snapshot_ref = &snapshot;
            let walker_ref = &walker;
            let arm_closures: Vec<_> = entries_ref
                .iter()
                .map(|&entry| {
                    move |arm_rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                        let mut n2v = FxHashMap::default();
                        for (i, (name, _)) in snapshot_ref.iter().enumerate() {
                            n2v.insert(name.clone(), arm_rb.param(i as u32));
                        }
                        let mut arm = RegionLowerer::new_child(arm_rb, fn_ctx, n2v);
                        let (st, vals) = multi_walk(&mut arm, state, entry, None, walker_ref)?;
                        Ok(BranchResult {
                            state: st,
                            values: vals,
                        })
                    }
                })
                .collect();
            let branch_refs = as_branch_refs(&arm_closures);
            let live_in_vals: Vec<ValueId> = snapshot.iter().map(|(_, v)| *v).collect();
            let gamma = body_rb.gamma_n(q_entry, state, &live_in_vals, &branch_refs)?;

            let total = base + 3; // base slots + q_entry + q_exit + r
            let values: Vec<ValueId> = (0..total as u16).map(|i| gamma.result(i)).collect();
            let next_vars: Vec<ValueId> = values[..base + 2].to_vec();
            Ok(LoopResult {
                condition: values[base + 2],
                next_state: gamma.state,
                next_vars,
            })
        })?;

        // Bind loop-closed values to their theta projections for the outer
        // scope.
        for (k, name) in closed_extras.iter().enumerate() {
            self.name_to_value
                .insert(name.clone(), theta_result.result((n_entry_phi + k) as u16));
        }

        // Post-theta exit dispatch on q_exit.
        let q_exit_value = theta_result.result((base + 1) as u16);
        let exit_targets: SmallVec<[BasicBlockId; 4]> =
            exit_arcs.iter().map(|&(_, dst)| dst).collect();
        if exit_targets.len() == 1 {
            Ok((theta_result.state, exit_targets[0]))
        } else {
            let join = self.compute_exit_targets_join(&exit_targets)?;
            let empty: Vec<Vec<(Name, ValueId)>> =
                exit_targets.iter().map(|_| Vec::new()).collect();
            let state_after = self.lower_exit_dispatch(
                theta_result.state,
                q_exit_value,
                &exit_targets,
                join,
                &empty,
            )?;
            Ok((state_after, join))
        }
    }

    /// Find the join block of a multi-exit theta dispatch: the deepest
    /// block in the post-dominator tree that post-dominates every exit
    /// target (their lowest common ancestor in that tree). An exit
    /// target can itself be the join, which happens when one of the
    /// exit paths converges immediately while the others step through
    /// intermediate blocks first; the simpler "every target's
    /// immediate post-dominator must match" check would miss this and
    /// report a spurious mismatch.
    ///
    /// Bails when one exit target's post-dominator chain does not
    /// contain another's: that means control along the two paths never
    /// reconverges inside this region, which needs the
    /// not-yet-implemented continuation-predicate transform (Bahmann
    /// et al. 2015 section 4.2's auxiliary p).
    fn compute_exit_targets_join(
        &self,
        exit_targets: &[BasicBlockId],
    ) -> color_eyre::Result<BasicBlockId> {
        // Only exit targets that can reach the function exit have a
        // continuation to reconverge at. An exit-unreachable target — one
        // that traps (`unreachable`) or otherwise never returns, so it has
        // no post-dominator — has no continuation, so it is excluded from
        // the join: the dispatch still routes to it, but it need not meet
        // the others. Without this, a returning exit and a trapping exit
        // share no post-dominator and `post_dominator_lca` bails.
        let mut reachable = exit_targets
            .iter()
            .copied()
            .filter(|&t| self.fn_ctx.post_immediate_dominators[t.0 as usize].is_some());
        let Some(first) = reachable.next() else {
            // Every exit traps; the loop has no live continuation. Resume
            // at the synthetic function exit (the resume point is dead).
            return Ok(self.fn_ctx.exit_block_id);
        };
        let mut join = first;
        for target in reachable {
            join = self.post_dominator_lca(join, target)?;
        }
        Ok(join)
    }

    /// Lowest common ancestor of `a` and `b` in the post-dominator
    /// tree. Walks `a`'s chain into a set, then walks `b`'s chain
    /// until it hits a block in that set. The chain for a block `x`
    /// starts at `x` itself (so when one exit target post-dominates
    /// another, the dominator is returned, not its parent).
    ///
    /// `compute_dominance` marks the start node's immediate dominator
    /// as itself (the post-dominator-tree root is the exit block,
    /// whose idom self-references); the chain walk uses the set's
    /// insert-return to break out the moment the same block reappears,
    /// rather than spinning forever at that root.
    fn post_dominator_lca(
        &self,
        a: BasicBlockId,
        b: BasicBlockId,
    ) -> color_eyre::Result<BasicBlockId> {
        let mut a_chain: FxHashSet<BasicBlockId> = FxHashSet::default();
        let mut cursor = Some(a);
        while let Some(block) = cursor {
            if !a_chain.insert(block) {
                break;
            }
            cursor = self.fn_ctx.post_immediate_dominators[block.0 as usize];
        }
        let mut cursor = Some(b);
        let mut b_seen: FxHashSet<BasicBlockId> = FxHashSet::default();
        while let Some(block) = cursor {
            if a_chain.contains(&block) {
                return Ok(block);
            }
            if !b_seen.insert(block) {
                break;
            }
            cursor = self.fn_ctx.post_immediate_dominators[block.0 as usize];
        }
        bail!(
            "multi-exit loop's exit targets {} and {} have no common \
             post-dominator: control along the two paths never \
             reconverges inside this region. Needs the not-yet-\
             implemented continuation-predicate transform.",
            a.0,
            b.0
        );
    }

    /// Build a post-theta gamma that dispatches on the q value to the
    /// correct exit target. Each arm walks one exit target up to the
    /// shared join, with the exit target's loop-closed phis pre-bound
    /// to their theta projection values.
    fn lower_exit_dispatch(
        &mut self,
        state: State,
        q_value: ValueId,
        exit_targets: &[BasicBlockId],
        join: BasicBlockId,
        lcssa_per_exit: &[Vec<(Name, ValueId)>],
    ) -> color_eyre::Result<State> {
        let phis_at_join = phi_instructions_at(&self.fn_ctx.func.basic_blocks[join.0 as usize]);

        let arm_blocks_per_arm: Vec<FxHashSet<BasicBlockId>> = exit_targets
            .iter()
            .map(|&target| self.collect_walked_blocks(target, &[join]))
            .collect();

        let combined_arm_blocks: SmallVec<[BasicBlockId; 8]> = arm_blocks_per_arm
            .iter()
            .flat_map(|set| set.iter().copied())
            .collect();
        let (standard_live_in_names, mut all_live_in_values) =
            self.compute_arm_live_ins(&combined_arm_blocks, &phis_at_join, None);

        // Per-exit loop-closed phi values become additional live-ins.
        // Record where each exit's block starts so the arm closure can
        // map names back to the right arm-param index.
        let mut lcssa_start_per_exit: Vec<usize> = Vec::with_capacity(exit_targets.len());
        for lcssas in lcssa_per_exit {
            lcssa_start_per_exit.push(all_live_in_values.len());
            for (_, value) in lcssas {
                all_live_in_values.push(*value);
            }
        }

        // Types of the join phis, used to synthesise poison results for any
        // exit-unreachable (trapping) arm so all arms share a signature.
        let phi_types: Vec<TypeRef> = phis_at_join
            .iter()
            .map(|p| {
                self.rb
                    .graph
                    .types
                    .convert_type_ref(&p.to_type, self.fn_ctx.llvm_mod)
            })
            .collect::<color_eyre::Result<Vec<_>>>()?;

        let fn_ctx = self.fn_ctx;
        let phis_slice: &[&Phi] = &phis_at_join;
        let phi_types_slice: &[TypeRef] = &phi_types;
        let standard_live_in_names_slice: &[Name] = &standard_live_in_names;

        let arm_closures: Vec<_> = exit_targets
            .iter()
            .enumerate()
            .map(|(arm_idx, &target)| {
                let lcssa_start = lcssa_start_per_exit[arm_idx];
                let lcssas_for_arm = &lcssa_per_exit[arm_idx];
                move |rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                    // An exit-unreachable target traps and never reaches the
                    // join. Emit poison for each join phi so this arm matches
                    // the others' signature; it is dead at runtime (taking
                    // this exit is undefined behaviour).
                    if fn_ctx.post_immediate_dominators[target.0 as usize].is_none() {
                        let values = phi_types_slice
                            .iter()
                            .map(|&ty| rb.constant(ty, ConstValue::Poison))
                            .collect();
                        return Ok(BranchResult { state, values });
                    }
                    let mut name_to_value = FxHashMap::with_capacity_and_hasher(
                        standard_live_in_names_slice.len() + lcssas_for_arm.len() + 4,
                        Default::default(),
                    );
                    for (i, name) in standard_live_in_names_slice.iter().enumerate() {
                        name_to_value.insert(name.clone(), rb.param(i as u32));
                    }
                    // Bind this exit's loop-closed dests to the
                    // corresponding live-in params. Other exits'
                    // bindings are also live-ins (so the gamma has the
                    // right input count) but their names are not
                    // referenced by this arm's walk.
                    for (i, (name, _)) in lcssas_for_arm.iter().enumerate() {
                        name_to_value.insert(name.clone(), rb.param((lcssa_start + i) as u32));
                    }

                    let mut arm = RegionLowerer::new_child(rb, fn_ctx, name_to_value);

                    // Exit-target loop-closed phis were already bound above,
                    // so the walk starts with no linear predecessor.
                    let (arm_state, exit_pred) =
                        match arm.lower_region(state, target, &[join], None, None)? {
                            RegionExit::AtBoundary {
                                state: s,
                                exit_pred,
                                ..
                            } => (s, exit_pred),
                            RegionExit::Returned { .. } => {
                                return Err(eyre!("early return inside multi-exit dispatch arm"));
                            }
                        };

                    let values = arm.arm_phi_contributions(phis_slice, exit_pred)?;
                    Ok(BranchResult {
                        state: arm_state,
                        values,
                    })
                }
            })
            .collect();

        let branch_refs = as_branch_refs(&arm_closures);
        // `q` is the i32 exit-arc index; the gamma routes value `k` to arm
        // `k`. The backend lowers an integer-conditioned gamma to a switch
        // (default = arm 0) regardless of arm count, so q feeds the gamma
        // directly — no need to special-case two exits into a boolean.
        let result = self
            .rb
            .gamma_n(q_value, state, &all_live_in_values, &branch_refs)?;

        for (i, phi) in phis_at_join.iter().enumerate() {
            self.name_to_value
                .insert(phi.dest.clone(), result.result(i as u16));
        }

        Ok(result.state)
    }

    /// Build a placeholder zero-valued constant of the given RVSDG
    /// type. Used as the init for loop-closed-extra loop_var slots,
    /// whose input value is overwritten each iteration via next_vars.
    fn zero_of(&mut self, ty: TypeRef) -> color_eyre::Result<ValueId> {
        match ty {
            TypeRef::Scalar(scalar) => match scalar {
                ScalarType::Bool
                | ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::I128 => Ok(self.rb.constant(ty, ConstValue::Int(0))),
                ScalarType::F32 => Ok(self.rb.constant(ty, ConstValue::F32(0))),
                ScalarType::F64 => Ok(self.rb.constant(ty, ConstValue::F64(0))),
                ScalarType::Void => bail!("cannot build a zero value of type Void"),
            },
            TypeRef::Ptr(_) => Ok(self.rb.constant(ty, ConstValue::NullPtr)),
            other => bail!(
                "zero_of: aggregate/vector loop-closed placeholder not yet supported \
                 (type: {:?})",
                other
            ),
        }
    }

    /// Analyse a strongly connected component to build the
    /// `LoopLowerCtx` that `lower_scc_as_theta` will consume. Pure
    /// analysis, except that it may emit constant values into the
    /// builder (for loop-closed extra placeholder inits and for
    /// constant-resolved header phi inits); these are outer-scope
    /// values, valid before the theta is built.
    pub(super) fn analyze_loop(
        &mut self,
        scc_id: SccTreeNodeId,
        header: BasicBlockId,
        exit_arcs: &[(BasicBlockId, BasicBlockId)],
    ) -> color_eyre::Result<LoopLowerCtx<'m>> {
        let body_blocks: SmallVec<[BasicBlockId; 8]> = self.fn_ctx.scc_tree.blocks
            [scc_id.0 as usize]
            .iter()
            .copied()
            .collect();

        // Enumerate header phis. Each has exactly two incomings for a
        // single-back-edge natural loop: preheader (outside body) and
        // back-edge (inside body, from the latch).
        let header_bb = &self.fn_ctx.func.basic_blocks[header.0 as usize];
        let header_phi_refs = phi_instructions_at(header_bb);
        let mut header_phis: SmallVec<[HeaderPhiInfo<'m>; 4]> = SmallVec::new();
        for phi in header_phi_refs.iter() {
            let mut init: Option<ValueId> = None;
            let mut next_operand: Option<&'m Operand> = None;
            for (op, pred_name) in &phi.incoming_values {
                let pred_id = *self.fn_ctx.bb_mapper.get(pred_name).ok_or_else(|| {
                    eyre!(
                        "header phi {:?} predecessor name {:?} unknown to bb_mapper",
                        phi.dest,
                        pred_name
                    )
                })?;
                if body_blocks.contains(&pred_id) {
                    next_operand = Some(op);
                } else {
                    init = Some(self.operand(op)?);
                }
            }
            let init =
                init.ok_or_else(|| eyre!("header phi {:?} has no preheader incoming", phi.dest))?;
            let next_operand = next_operand.ok_or_else(|| {
                eyre!(
                    "header phi {:?} has no in-body (back-edge) incoming",
                    phi.dest
                )
            })?;
            header_phis.push(HeaderPhiInfo {
                phi,
                init,
                next_operand,
            });
        }

        // Classify loop-closed phis at each exit arc's destination.
        // For each exit arc (src, dst): scan dst's phis; for each phi
        // that has an incoming from `src`, classify the operand
        // (sub-case A / B / C). Record per-exit.
        let mut lcssa_bindings_per_exit: SmallVec<[SmallVec<[(Name, LcssaBinding); 2]>; 4]> =
            SmallVec::with_capacity(exit_arcs.len());
        let mut lcssa_extras: SmallVec<[LcssaExtraSlot; 2]> = SmallVec::new();
        let mut lcssa_extra_inits: SmallVec<[ValueId; 2]> = SmallVec::new();

        for &(exit_src, exit_dst) in exit_arcs {
            let mut bindings: SmallVec<[(Name, LcssaBinding); 2]> = SmallVec::new();
            let exit_bb = &self.fn_ctx.func.basic_blocks[exit_dst.0 as usize];
            let phis_at_exit = phi_instructions_at(exit_bb);

            for lcssa_phi in phis_at_exit.iter() {
                // Find the incoming for THIS exit arc (whose
                // predecessor is `exit_src`). Other incomings belong
                // to other exit arcs; they'll be classified when we
                // process those.
                let incoming =
                    phi_incoming_from(lcssa_phi, self.fn_ctx.bb_mapper, |id| id == exit_src);
                let (op, _) = match incoming {
                    Some(p) => p,
                    None => continue,
                };

                let binding = match op {
                    Operand::ConstantOperand(_) => {
                        let v = self.operand(op)?;
                        LcssaBinding::Constant(v)
                    }
                    Operand::LocalOperand { name, ty: llvm_ty } => {
                        let back_edge_match = header_phis.iter().position(|info| {
                            matches!(
                                info.next_operand,
                                Operand::LocalOperand { name: n, .. } if n == name
                            )
                        });
                        let header_dest_match =
                            header_phis.iter().position(|info| &info.phi.dest == name);
                        if let Some(idx) = back_edge_match {
                            // Sub-case A: the loop-closed phi references a
                            // header phi's back-edge operand; the slot's
                            // post-loop projection is exactly that value.
                            LcssaBinding::HeaderPhi { index: idx as u16 }
                        } else if let Some(idx) = header_dest_match.filter(|_| exit_src == header) {
                            // Sub-case B at the natural (header) exit: no
                            // body work has executed, so the slot still
                            // holds the header phi's input value, which is
                            // exactly the slot's post-loop projection. Fast
                            // path that avoids allocating an extra slot.
                            LcssaBinding::HeaderPhi { index: idx as u16 }
                        } else {
                            // One of:
                            //   - a body-internal value used after the loop, or
                            //   - sub-case B at a NON-header exit: the
                            //     loop-closed phi references a header phi
                            //     dest but the exit leaves mid-body.
                            //
                            // Both thread out through a dedicated extra
                            // loop_var slot that captures the CURRENT value
                            // of `name` at the exiting iteration's leaf. For
                            // a header phi dest, `name` is never reassigned
                            // inside the body (it is an SSA def at the
                            // header), so the slot captures the pre-update
                            // header value - precisely what the loop-closed
                            // phi wants, regardless of whether the back-edge
                            // operand happens to be defined before the exit.
                            // This is the demand-analysis binding that
                            // replaces the previous bail for non-natural
                            // exits. The body-seeding step in
                            // `lower_scc_as_theta` skips re-binding names
                            // already bound as header params, so the header
                            // binding the body relies on is preserved.
                            let rvsdg_ty = self
                                .rb
                                .graph
                                .types
                                .convert_type_ref(llvm_ty, self.fn_ctx.llvm_mod)?;
                            let init_id = self.zero_of(rvsdg_ty)?;
                            // Reuse an existing extra slot if this name
                            // already has one (two exits' loop-closed phis
                            // can reference the same value).
                            let extra_idx =
                                match lcssa_extras.iter().position(|slot| slot.name == *name) {
                                    Some(idx) => idx as u16,
                                    None => {
                                        let idx = lcssa_extras.len() as u16;
                                        lcssa_extras.push(LcssaExtraSlot {
                                            name: name.clone(),
                                            ty: rvsdg_ty,
                                        });
                                        lcssa_extra_inits.push(init_id);
                                        idx
                                    }
                                };
                            LcssaBinding::Extra { index: extra_idx }
                        }
                    }
                    Operand::MetadataOperand => {
                        bail!(
                            "loop-closed phi {:?} has a metadata operand",
                            lcssa_phi.dest
                        )
                    }
                };
                bindings.push((lcssa_phi.dest.clone(), binding));
            }
            lcssa_bindings_per_exit.push(bindings);
        }

        // Live-in scan: walk every body block, accumulate names used
        // inside but defined outside. Header phis are skipped (their
        // preheader operand is already captured in `init` above;
        // walking the phi here would add a redundant live-in slot).
        let mut defined_inside: FxHashSet<&Name> = FxHashSet::default();
        let mut body_instr_count: u32 = 0;
        for &bb_id in &body_blocks {
            let bb = &self.fn_ctx.func.basic_blocks[bb_id.0 as usize];
            body_instr_count += bb.instrs.len() as u32;
            for inst in &bb.instrs {
                if let Some(dest) = instruction_dest(inst) {
                    defined_inside.insert(dest);
                }
            }
        }

        let mut seen: FxHashSet<&'m Name> = FxHashSet::default();
        let mut live_in_names: SmallVec<[Name; 4]> = SmallVec::new();
        let mut live_in_inits: SmallVec<[ValueId; 4]> = SmallVec::new();
        let name_to_value = &self.name_to_value;
        let defined_inside_ref = &defined_inside;

        let visit = |op: &'m Operand,
                     seen: &mut FxHashSet<&'m Name>,
                     live_in_names: &mut SmallVec<[Name; 4]>,
                     live_in_inits: &mut SmallVec<[ValueId; 4]>| {
            let Operand::LocalOperand { name, .. } = op else {
                return;
            };
            if defined_inside_ref.contains(name) || !seen.insert(name) {
                return;
            }
            if let Some(&val) = name_to_value.get(name) {
                live_in_names.push(name.clone());
                live_in_inits.push(val);
            }
        };

        for &bb_id in &body_blocks {
            let bb = &self.fn_ctx.func.basic_blocks[bb_id.0 as usize];
            for inst in &bb.instrs {
                if bb_id == header && matches!(inst, Instruction::Phi(_)) {
                    continue;
                }
                for_each_operand(inst, |op| {
                    visit(op, &mut seen, &mut live_in_names, &mut live_in_inits)
                });
            }
            // Terminator operands can reference outer-scope values too.
            // Clang typically routes a branch condition through an
            // in-body icmp whose result lives in bb.instrs, but a direct
            // branch on a function argument or other purely outer SSA
            // value goes through bb.term alone. Pick those up here so
            // the value becomes a theta live-in and is reachable from
            // body.name_to_value when the walker resolves the condition.
            for_each_terminator_operand(&bb.term, |op| {
                visit(op, &mut seen, &mut live_in_names, &mut live_in_inits)
            });
        }

        // Build the LoopVarInits buffer.
        let mut storage: SmallVec<[ValueId; 16]> = SmallVec::new();
        storage.extend(header_phis.iter().map(|p| p.init));
        let header_phi_end = storage.len() as u16;
        storage.extend(live_in_inits.iter().copied());
        let live_in_end = storage.len() as u16;
        storage.extend(lcssa_extra_inits.iter().copied());
        let inits = LoopVarInits {
            storage,
            header_phi_end,
            live_in_end,
        };

        Ok(LoopLowerCtx {
            body_instr_count,
            inits,
            header_phis,
            live_in_names,
            lcssa_extras,
            lcssa_bindings_per_exit,
        })
    }
}

// ============================================================================
// Body walker: recursive gamma tree.
// ============================================================================

/// Context passed to the recursive body walker. All fields are
/// borrowed; the walker doesn't own any of them.
struct BodyWalker<'a, 'm> {
    /// The loop's entry vertex. Targets equal to `header` mark
    /// repetition arcs (back-edge to the loop's start).
    header: BasicBlockId,
    /// All blocks belonging to this SCC's body. A successor target
    /// outside this slice is on an exit arc. A slice rather than a
    /// hashset because SCC bodies are small (well under 16 blocks
    /// for typical loops) and `slice::contains` beats a hash lookup
    /// at that size while avoiding a per-SCC allocation.
    scc_body: &'a [BasicBlockId],
    /// Exit arcs in dispatch-index order: `exit_arc_indices[k] = (src, dst)`
    /// of exit arc `k`. The walker linearly scans this for the index
    /// of an exit arc at every exit leaf; with 1-4 exits per loop in
    /// practice, the scan is cheaper than a hashmap lookup.
    exit_arc_indices: &'a [(BasicBlockId, BasicBlockId)],
    /// Analysis context: header phis, live-in names, loop-closed extra
    /// names.
    ctx: &'a LoopLowerCtx<'m>,
    /// Total theta loop_var slots: the sum of header_phis, live_in_names,
    /// and lcssa_extras in `ctx`. Each leaf produces `slot_count + 2`
    /// values (slots, q, r).
    slot_count: u16,
}

/// Walk a single block in the loop body. Recurses through unconditional
/// branches and builds gamma trees at conditional branches. Returns the
/// post-walk state and the leaf's (or merged leaves') value tuple.
///
/// `prev` is the predecessor block we arrived from, or `None` at the
/// top-level call (the start of an iteration at the loop header). It
/// is used to resolve interior phi nodes path-aware: when an arm of
/// an outer gamma reaches a join block inside the loop body, the join
/// block's phi destinations are bound to whichever incoming
/// corresponds to the predecessor we came in through. Without this,
/// subsequent instructions in the join block that use the phi's
/// destination panic in `operand()` because the phi was skipped by
/// `lower_instructions_skip_phis`.
/// If the branch terminating `block` reconverges inside the loop body,
/// return that join. The join is the block's post-dominator when it lies
/// within the SCC body and is not the header: the arms then form an
/// acyclic single-join branch of L* (the body minus the repetition arc),
/// which Bahmann et al. §4.1 lowers by recursing into §4.2 branch
/// restructuring. The join is lowered exactly once.
///
/// `None` when any arm is instead a leaf — a repetition (target is the
/// header) or an exit/trap (target leaves the SCC body, e.g. a `return`
/// or `unreachable` block) — which the body walk handles directly. This
/// guard is necessary because post-dominance *ignores* paths that cannot
/// reach the function exit (trapping `unreachable` arms get no post-idom),
/// so a mixed branch's post-idom can be an in-body block even though one
/// arm never reaches it (fixture 35). Requiring every arm target to be a
/// non-header body block rules that out: post-dominance then guarantees
/// every arm reaches the join before looping or exiting, so the arm walk
/// stays inside the body.
fn in_body_join(
    body: &RegionLowerer<'_, '_, '_>,
    block: BasicBlockId,
    walker: &BodyWalker<'_, '_>,
) -> Option<BasicBlockId> {
    let join = body.fn_ctx.post_immediate_dominators[block.0 as usize]?;
    if join == walker.header || !walker.scc_body.contains(&join) {
        return None;
    }
    let all_arms_reconverge = body
        .fn_ctx
        .bb_mapper
        .outputs(block)
        .iter()
        .all(|target| *target != walker.header && walker.scc_body.contains(target));
    all_arms_reconverge.then_some(join)
}

fn lower_body_walk<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    prev: Option<BasicBlockId>,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let fn_ctx = body.fn_ctx;

    // If this block is the entry of an inner SCC (and we are not at
    // our own loop's header, which has its own SCC entry marker but
    // is processed by the outer walker, not as an inner loop),
    // dispatch into the inner loop and continue at its exit target.
    //
    // The inner SCC's exit must land back inside the outer SCC's body
    // (or at the outer header for an immediate continue). When an
    // inner exit goes straight outside the outer SCC, the true exit
    // arc source is a block inside the inner SCC and the outer
    // walker has no way to recover it: walk_target would search
    // exit_arc_indices for the pair (inner_scc_header, inner_exit),
    // which is never recorded because the SCC analysis stored the
    // arc at the inner-internal block where the branch actually
    // happened. Detect and reject this shape rather than miscompile.
    if block != walker.header {
        if let Some(inner_scc_id) = fn_ctx.scc_entry_block_to_id[block.0 as usize] {
            let (next_state, inner_exit) = body.lower_scc_as_theta(state, inner_scc_id, block)?;
            if inner_exit != walker.header && !walker.scc_body.contains(&inner_exit) {
                bail!(
                    "loop body at block {} contains an inner loop whose exit \
                     target ({}) lies outside the outer loop body. This is the \
                     break-out-of-nested-loop shape: the actual exit arc source \
                     is a block inside the inner loop, so the outer walker \
                     cannot identify which exit arc index to emit. Supporting \
                     this requires the inner loop's exit dispatch to also \
                     write into the outer theta's q slot, which is not yet \
                     implemented.",
                    block.0,
                    inner_exit.0,
                );
            }
            return walk_target(body, next_state, block, inner_exit, walker);
        }
    }

    // Resolve interior-join phis from the arm-path predecessor.
    //
    // Each arm of an enclosing gamma walks the body independently, so
    // two arms that converge at an interior block J both visit J and
    // lower its instructions. The phi at J selects between values
    // produced along the two predecessor paths; inside each arm we
    // know exactly which predecessor we came in through (`prev`), so
    // we can bind J's phi destinations directly to that incoming's
    // value. Subsequent instructions in J that reference the phi
    // destination resolve via the binding.
    //
    // The cost of this path-aware approach is that instructions in J
    // and any blocks the arms continue walking through together get
    // lowered once per arm rather than once via a gamma node whose
    // join sits at J. A future optimisation pass over the body could
    // detect that and fold duplicate sub-DAGs, but correctness is the
    // priority here; the alternative (recursing into the construction
    // algorithm's branch-restructuring transform for the body's
    // acyclic subgraph) needs much more machinery to land.
    //
    // For the loop header (top-level call, prev = None), phis are
    // pre-seeded as theta loop_var params and must not be rebound.
    // For inner-loop exit blocks reached after an inner-SCC dispatch,
    // the inner SCC's `lower_scc_as_theta` has already bound any
    // loop-closed phi destinations into `name_to_value`; those phis
    // have no incoming from `prev` (which is the inner SCC's entry
    // vertex from our perspective, not an in-inner-loop predecessor),
    // so `phi_incoming_from` returns None and we leave the binding
    // alone.
    if let Some(pred) = prev {
        let bb = &fn_ctx.func.basic_blocks[block.0 as usize];
        let phis = phi_instructions_at(bb);
        for phi in &phis {
            if let Some((op, _)) = phi_incoming_from(phi, fn_ctx.bb_mapper, |id| id == pred) {
                let value = body.operand(op)?;
                body.name_to_value.insert(phi.dest.clone(), value);
            }
        }
    }

    let state = body.lower_instructions_skip_phis(state, block)?;

    let bb = &fn_ctx.func.basic_blocks[block.0 as usize];
    match &bb.term {
        llvm_ir::Terminator::Br(br) => {
            let target = *fn_ctx.bb_mapper.get_expect(&br.dest);
            walk_target(body, state, block, target, walker)
        }
        llvm_ir::Terminator::CondBr(cond_br) => {
            // If the arms reconverge at a block inside the loop body, this
            // is an ordinary single-join branch of L* — the loop body minus
            // the repetition arc, which Bahmann et al. §4.1 hands to the
            // §4.2 branch restructuring. Lower it with the shared acyclic
            // gamma machinery, which walks each arm only as far as the join
            // and lowers the join (and everything after it) exactly once,
            // then resume the body walk from the join. This is what keeps
            // body lowering linear: the leaf walk below instead re-walks the
            // whole post-join suffix once per arm, which is exponential on a
            // chain of reconverging branches (and overflows the stack).
            if let Some(join) = in_body_join(body, block, walker) {
                let state = body.lower_cond_branch(state, cond_br, block, join)?;
                return lower_body_walk(body, state, join, None, walker);
            }
            let cond_value = body.operand(&cond_br.condition)?;
            let true_target = *fn_ctx.bb_mapper.get_expect(&cond_br.true_dest);
            let false_target = *fn_ctx.bb_mapper.get_expect(&cond_br.false_dest);
            lower_body_cond_branch(
                body,
                state,
                block,
                cond_value,
                true_target,
                false_target,
                walker,
            )
        }
        llvm_ir::Terminator::Switch(switch) => {
            // Same single-join shortcut as CondBr (see above).
            if let Some(join) = in_body_join(body, block, walker) {
                let state = body.lower_switch(state, switch, block, join)?;
                return lower_body_walk(body, state, join, None, walker);
            }
            lower_body_switch(body, state, block, switch, walker)
        }
        llvm_ir::Terminator::Ret(_) => {
            bail!(
                "early return inside loop body (block {}) not supported",
                block.0
            )
        }
        llvm_ir::Terminator::Unreachable(_) => bail!(
            "unreachable terminator inside loop body (block {})",
            block.0
        ),
        other => bail!("unsupported terminator inside loop body: {:?}", other),
    }
}

/// Resolve a target block to either a leaf (repetition arc or exit arc)
/// or a recursive walk (intra-SCC continuation). `src` is the block
/// whose terminator points at `target`; it is needed to look up the
/// exit-arc index when the target is outside the SCC.
fn walk_target<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    src: BasicBlockId,
    target: BasicBlockId,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    if target == walker.header {
        let values = make_rep_leaf(body, walker)?;
        return Ok((state, values));
    }
    if !walker.scc_body.contains(&target) {
        let k = walker
            .exit_arc_indices
            .iter()
            .position(|&arc| arc == (src, target))
            .ok_or_else(|| {
                eyre!(
                    "exit arc ({}, {}) not in exit_arc_indices; \
                     SCC analysis out of sync",
                    src.0,
                    target.0
                )
            })? as u32;
        let values = make_exit_leaf(body, k, walker)?;
        return Ok((state, values));
    }
    lower_body_walk(body, state, target, Some(src), walker)
}

/// Build an n-arm gamma at an in-body branch. Snapshots the body's
/// `name_to_value` as live-ins so each arm's region can resolve SSA names
/// defined along the path leading up to the branch, then builds one arm
/// per `target` (each continues the walk via `walk_target`) selected by
/// `selector`. Shared by the conditional-branch and switch lowering below,
/// which differ only in how they compute `selector`/`targets`.
fn lower_body_n_way<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    selector: ValueId,
    targets: &[BasicBlockId],
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let fn_ctx = body.fn_ctx;

    // Snapshot the body's `name_to_value` as parallel name-pointer
    // and value arrays. Borrowing the names avoids cloning every
    // Name (each clone is a heap allocation, since Name::Name wraps
    // Box<String>) into a temporary pairs vector; the only clones
    // that survive are those that end up in each arm's permanent
    // name_to_value map.
    let mut live_in_names: Vec<&Name> = Vec::with_capacity(body.name_to_value.len());
    let mut live_in_values: Vec<ValueId> = Vec::with_capacity(body.name_to_value.len());
    for (name, &value) in &body.name_to_value {
        live_in_names.push(name);
        live_in_values.push(value);
    }

    let walker_ref = walker;
    let names_ref = &live_in_names;
    let block_src = block;

    let arm_closures: Vec<_> = targets
        .iter()
        .map(|&target| {
            move |arm_rb: &mut RegionBuilder| -> color_eyre::Result<BranchResult> {
                let mut arm_n2v =
                    FxHashMap::with_capacity_and_hasher(names_ref.len() + 4, Default::default());
                for (i, name) in names_ref.iter().enumerate() {
                    arm_n2v.insert((*name).clone(), arm_rb.param(i as u32));
                }
                let mut arm = RegionLowerer::new_child(arm_rb, fn_ctx, arm_n2v);
                let (arm_state, values) =
                    walk_target(&mut arm, state, block_src, target, walker_ref)?;
                Ok(BranchResult {
                    state: arm_state,
                    values,
                })
            }
        })
        .collect();
    let branch_refs = as_branch_refs(&arm_closures);

    let gamma = body
        .rb
        .gamma_n(selector, state, &live_in_values, &branch_refs)?;

    let total_outputs = walker.slot_count as usize + 2;
    let values: Vec<ValueId> = (0..total_outputs as u16).map(|i| gamma.result(i)).collect();
    Ok((gamma.state, values))
}

/// Conditional branch inside the body: a two-arm gamma (arm 0 = true
/// target, arm 1 = false target).
fn lower_body_cond_branch<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    cond_value: ValueId,
    true_target: BasicBlockId,
    false_target: BasicBlockId,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    lower_body_n_way(
        body,
        state,
        block,
        cond_value,
        &[true_target, false_target],
        walker,
    )
}

/// Switch inside the body: an n-arm gamma. Reuses the same arm-index
/// selector as the acyclic switch lowering (`RegionLowerer::switch_selector`).
fn lower_body_switch<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    switch: &llvm_ir::terminator::Switch,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let (selector, targets) = body.switch_selector(switch)?;
    lower_body_n_way(body, state, block, selector, &targets, walker)
}

/// Build the slot values portion of a leaf tuple. Each header phi
/// slot resolves to the back-edge operand if it has been defined along
/// this path; otherwise it falls back to the slot's "current value"
/// (the header phi destination, which is always bound to the slot's
/// input via the seeding done at the top of `lower_scc_as_theta`).
/// Live-in and loop-closed extra slots resolve directly via the
/// always-present name bindings.
fn make_leaf_slot_values<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<Vec<ValueId>> {
    let mut values = Vec::with_capacity(walker.slot_count as usize + 2);

    for info in &walker.ctx.header_phis {
        match try_operand(body, info.next_operand)? {
            Some(v) => values.push(v),
            None => {
                let dest_value =
                    body.name_to_value
                        .get(&info.phi.dest)
                        .copied()
                        .ok_or_else(|| {
                            eyre!(
                                "header phi dest {:?} not in name_to_value at leaf",
                                info.phi.dest
                            )
                        })?;
                values.push(dest_value);
            }
        }
    }
    for name in &walker.ctx.live_in_names {
        let v = body
            .name_to_value
            .get(name)
            .copied()
            .ok_or_else(|| eyre!("live-in name {:?} not in name_to_value at leaf", name))?;
        values.push(v);
    }
    for slot in &walker.ctx.lcssa_extras {
        let v = body.name_to_value.get(&slot.name).copied().ok_or_else(|| {
            eyre!(
                "loop-closed extra name {:?} not in name_to_value at leaf",
                slot.name
            )
        })?;
        values.push(v);
    }
    Ok(values)
}

fn make_rep_leaf<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<Vec<ValueId>> {
    let mut values = make_leaf_slot_values(body, walker)?;
    // Repetition leaves do not need a meaningful q (q is read only when
    // r = 0). Pick 0 as a canonical placeholder.
    values.push(body.rb.const_i32(0));
    values.push(body.rb.constant(BOOL, ConstValue::Int(1)));
    Ok(values)
}

fn make_exit_leaf<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    exit_idx: u32,
    walker: &BodyWalker<'_, 'm>,
) -> color_eyre::Result<Vec<ValueId>> {
    let mut values = make_leaf_slot_values(body, walker)?;
    values.push(body.rb.const_i32(exit_idx as i32));
    values.push(body.rb.constant(BOOL, ConstValue::Int(0)));
    Ok(values)
}

/// Try to resolve an operand in the current region. Succeeds (returns
/// `Some`) for constants and for local operands whose name is in
/// `name_to_value`. Returns `None` (not an error) when the local name
/// has not yet been defined along the current path. The walker uses
/// `None` as a signal to fall back to the slot's current value.
fn try_operand<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    op: &Operand,
) -> color_eyre::Result<Option<ValueId>> {
    match op {
        Operand::ConstantOperand(_) => Ok(Some(body.operand(op)?)),
        Operand::LocalOperand { name, .. } => {
            if body.name_to_value.contains_key(name) {
                Ok(Some(body.operand(op)?))
            } else {
                Ok(None)
            }
        }
        Operand::MetadataOperand => Ok(None),
    }
}

// ============================================================================
// Multi-entry (irreducible) body walker: the paper's §4.1 `q` entry dispatch.
// ============================================================================

/// One phi at an entry vertex of a multi-entry SCC. Each becomes a theta
/// loop_var carried across the `q` entry dispatch.
struct EntryPhi<'m> {
    entry: BasicBlockId,
    phi: &'m Phi,
}

/// Context for the multi-entry body walk. A leaf produces the tuple
/// `[entry_phis..., closed..., q_entry, q_exit, r]` — `base + 3` values
/// where `base = entry_phis.len() + closed.len()`.
struct MultiWalker<'a, 'm> {
    entries: &'a [BasicBlockId],
    scc_body: &'a [BasicBlockId],
    entry_phis: &'a [EntryPhi<'m>],
    closed: &'a [Name],
    exit_arcs: &'a [(BasicBlockId, BasicBlockId)],
    base: usize,
}

/// Walk one entry vertex's region inside an irreducible loop body.
/// Recurses through `Br`/`CondBr`; a target that is an entry vertex is a
/// repetition (sets `q_entry` to that entry), a target outside the SCC is
/// an exit (sets `q_exit`).
fn multi_walk<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    prev: Option<BasicBlockId>,
    walker: &MultiWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let fn_ctx = body.fn_ctx;

    // A nested inner loop inside an irreducible loop is not modelled yet.
    if !walker.entries.contains(&block) {
        if fn_ctx.scc_entry_block_to_id[block.0 as usize].is_some() {
            bail!(
                "nested loop inside an irreducible loop is not yet supported (block {})",
                block.0
            );
        }
    }

    // Interior-join phis resolve from the predecessor we arrived through.
    // Entry-vertex phis are loop_vars (seeded), so are not rebound here.
    if let Some(pred) = prev {
        if !walker.entries.contains(&block) {
            let bb = &fn_ctx.func.basic_blocks[block.0 as usize];
            for phi in &phi_instructions_at(bb) {
                if let Some((op, _)) = phi_incoming_from(phi, fn_ctx.bb_mapper, |id| id == pred) {
                    let v = body.operand(op)?;
                    body.name_to_value.insert(phi.dest.clone(), v);
                }
            }
        }
    }

    let state = body.lower_instructions_skip_phis(state, block)?;
    let bb = &fn_ctx.func.basic_blocks[block.0 as usize];
    match &bb.term {
        llvm_ir::Terminator::Br(br) => {
            let t = *fn_ctx.bb_mapper.get_expect(&br.dest);
            multi_walk_target(body, state, block, t, walker)
        }
        llvm_ir::Terminator::CondBr(cb) => {
            let cond = body.operand(&cb.condition)?;
            let tt = *fn_ctx.bb_mapper.get_expect(&cb.true_dest);
            let ft = *fn_ctx.bb_mapper.get_expect(&cb.false_dest);
            multi_cond_branch(body, state, block, cond, tt, ft, walker)
        }
        other => bail!(
            "unsupported terminator in irreducible loop body (block {}): {:?}",
            block.0,
            other
        ),
    }
}

fn multi_walk_target<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    src: BasicBlockId,
    target: BasicBlockId,
    walker: &MultiWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    if let Some(q) = walker.entries.iter().position(|&e| e == target) {
        let vals = make_multi_rep_leaf(body, src, target, q as i32, walker)?;
        return Ok((state, vals));
    }
    if !walker.scc_body.contains(&target) {
        let k = walker
            .exit_arcs
            .iter()
            .position(|&a| a == (src, target))
            .ok_or_else(|| eyre!("exit arc ({}, {}) not in SCC exit arcs", src.0, target.0))?
            as i32;
        let vals = make_multi_exit_leaf(body, k, walker)?;
        return Ok((state, vals));
    }
    multi_walk(body, state, target, Some(src), walker)
}

/// Two-way branch inside an irreducible loop body: a gamma whose arms each
/// continue the multi-entry walk. Mirrors `lower_body_cond_branch` but
/// produces the wider `base + 3` leaf tuple.
fn multi_cond_branch<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    cond: ValueId,
    true_target: BasicBlockId,
    false_target: BasicBlockId,
    walker: &MultiWalker<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let fn_ctx = body.fn_ctx;
    let mut names: Vec<&Name> = Vec::with_capacity(body.name_to_value.len());
    let mut vals: Vec<ValueId> = Vec::with_capacity(body.name_to_value.len());
    for (n, &v) in &body.name_to_value {
        names.push(n);
        vals.push(v);
    }
    let names_ref = &names;
    let walker_ref = walker;
    let src = block;

    let build = |arm_rb: &mut RegionBuilder,
                 target: BasicBlockId|
     -> color_eyre::Result<BranchResult> {
        let mut n2v = FxHashMap::with_capacity_and_hasher(names_ref.len() + 4, Default::default());
        for (i, n) in names_ref.iter().enumerate() {
            n2v.insert((*n).clone(), arm_rb.param(i as u32));
        }
        let mut arm = RegionLowerer::new_child(arm_rb, fn_ctx, n2v);
        let (st, v) = multi_walk_target(&mut arm, state, src, target, walker_ref)?;
        Ok(BranchResult {
            state: st,
            values: v,
        })
    };
    let ta = |rb: &mut RegionBuilder| build(rb, true_target);
    let fa = |rb: &mut RegionBuilder| build(rb, false_target);
    let gamma = body.rb.gamma(cond, state, &vals, ta, fa)?;

    let total = walker.base + 3;
    let out: Vec<ValueId> = (0..total as u16).map(|i| gamma.result(i)).collect();
    Ok((gamma.state, out))
}

/// Repetition leaf: control loops back to entry `rep_target` (q index
/// `q_idx`) from block `src`. The target entry's phis take their values
/// off that repetition arc; all other entry phis and closed values pass
/// through unchanged.
fn make_multi_rep_leaf<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    src: BasicBlockId,
    rep_target: BasicBlockId,
    q_idx: i32,
    walker: &MultiWalker<'_, 'm>,
) -> color_eyre::Result<Vec<ValueId>> {
    let mut vals: Vec<ValueId> = Vec::with_capacity(walker.base + 3);
    for ep in walker.entry_phis {
        if ep.entry == rep_target {
            let (op, _) = phi_incoming_from(ep.phi, body.fn_ctx.bb_mapper, |id| id == src)
                .ok_or_else(|| {
                    eyre!(
                        "entry phi {:?} has no incoming for repetition arc from block {}",
                        ep.phi.dest,
                        src.0
                    )
                })?;
            vals.push(body.operand(op)?);
        } else {
            vals.push(*body.name_to_value.get(&ep.phi.dest).ok_or_else(|| {
                eyre!("entry phi {:?} not bound at repetition leaf", ep.phi.dest)
            })?);
        }
    }
    for name in walker.closed {
        vals.push(
            *body.name_to_value.get(name).ok_or_else(|| {
                eyre!("loop-closed value {:?} not bound at repetition leaf", name)
            })?,
        );
    }
    vals.push(body.rb.const_i32(q_idx)); // q_entry
    vals.push(body.rb.const_i32(0)); // q_exit (unused on repetition)
    vals.push(body.rb.constant(BOOL, ConstValue::Int(1))); // r = repeat
    Ok(vals)
}

/// Exit leaf: control leaves the loop on exit-arc index `exit_idx`. Entry
/// phis and closed values pass through (their final values become theta
/// outputs the post-loop dispatch reads).
fn make_multi_exit_leaf<'m>(
    body: &mut RegionLowerer<'_, '_, 'm>,
    exit_idx: i32,
    walker: &MultiWalker<'_, 'm>,
) -> color_eyre::Result<Vec<ValueId>> {
    let mut vals: Vec<ValueId> = Vec::with_capacity(walker.base + 3);
    for ep in walker.entry_phis {
        vals.push(
            *body
                .name_to_value
                .get(&ep.phi.dest)
                .ok_or_else(|| eyre!("entry phi {:?} not bound at exit leaf", ep.phi.dest))?,
        );
    }
    for name in walker.closed {
        vals.push(
            *body
                .name_to_value
                .get(name)
                .ok_or_else(|| eyre!("loop-closed value {:?} not bound at exit leaf", name))?,
        );
    }
    vals.push(body.rb.const_i32(0)); // q_entry (unused on exit)
    vals.push(body.rb.const_i32(exit_idx)); // q_exit
    vals.push(body.rb.constant(BOOL, ConstValue::Int(0))); // r = exit
    Ok(vals)
}

/// Context for the entry-region walk that computes the initial `q` and the
/// entry-phi init values for an irreducible loop. Runs in the outer scope
/// (before the theta), producing the tuple `[q, init_0, .., init_{n-1}]`.
struct EntryCtx<'a, 'm> {
    entries: &'a [BasicBlockId],
    entry_phis: &'a [EntryPhi<'m>],
    #[allow(dead_code)]
    scc_body: &'a [BasicBlockId],
}

/// Walk the acyclic entry region from the dispatch dominator toward the
/// loop entries. Each path ends at a loop entry (a leaf yielding `q` plus
/// the entry-phi inits); a branch becomes a gamma that merges the tuples.
fn entry_walk<'m>(
    lowerer: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    prev: Option<BasicBlockId>,
    ctx: &EntryCtx<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let fn_ctx = lowerer.fn_ctx;

    if ctx.entries.contains(&block) {
        let src = prev.ok_or_else(|| {
            eyre!(
                "loop entry {} reached with no predecessor in entry region",
                block.0
            )
        })?;
        let vals = make_entry_leaf(lowerer, block, src, ctx)?;
        return Ok((state, vals));
    }

    // Interior-join phis in the entry region resolve from `prev`.
    if let Some(pred) = prev {
        let bb = &fn_ctx.func.basic_blocks[block.0 as usize];
        for phi in &phi_instructions_at(bb) {
            if let Some((op, _)) = phi_incoming_from(phi, fn_ctx.bb_mapper, |id| id == pred) {
                let v = lowerer.operand(op)?;
                lowerer.name_to_value.insert(phi.dest.clone(), v);
            }
        }
    }

    let state = lowerer.lower_instructions_skip_phis(state, block)?;
    let bb = &fn_ctx.func.basic_blocks[block.0 as usize];
    match &bb.term {
        llvm_ir::Terminator::Br(br) => {
            let t = *fn_ctx.bb_mapper.get_expect(&br.dest);
            entry_walk(lowerer, state, t, Some(block), ctx)
        }
        llvm_ir::Terminator::CondBr(cb) => {
            let cond = lowerer.operand(&cb.condition)?;
            let tt = *fn_ctx.bb_mapper.get_expect(&cb.true_dest);
            let ft = *fn_ctx.bb_mapper.get_expect(&cb.false_dest);
            entry_cond_branch(lowerer, state, block, cond, tt, ft, ctx)
        }
        other => bail!(
            "unsupported terminator in irreducible-loop entry region (block {}): {:?}",
            block.0,
            other
        ),
    }
}

/// Two-way branch in the entry region: a gamma whose arms each continue
/// the entry walk, merging their `[q, inits]` tuples.
fn entry_cond_branch<'m>(
    lowerer: &mut RegionLowerer<'_, '_, 'm>,
    state: State,
    block: BasicBlockId,
    cond: ValueId,
    true_target: BasicBlockId,
    false_target: BasicBlockId,
    ctx: &EntryCtx<'_, 'm>,
) -> color_eyre::Result<(State, Vec<ValueId>)> {
    let fn_ctx = lowerer.fn_ctx;
    let mut names: Vec<&Name> = Vec::with_capacity(lowerer.name_to_value.len());
    let mut vals: Vec<ValueId> = Vec::with_capacity(lowerer.name_to_value.len());
    for (n, &v) in &lowerer.name_to_value {
        names.push(n);
        vals.push(v);
    }
    let names_ref = &names;
    let ctx_ref = ctx;
    let src = block;

    let build = |arm_rb: &mut RegionBuilder,
                 target: BasicBlockId|
     -> color_eyre::Result<BranchResult> {
        let mut n2v = FxHashMap::with_capacity_and_hasher(names_ref.len() + 4, Default::default());
        for (i, n) in names_ref.iter().enumerate() {
            n2v.insert((*n).clone(), arm_rb.param(i as u32));
        }
        let mut arm = RegionLowerer::new_child(arm_rb, fn_ctx, n2v);
        let (st, v) = entry_walk(&mut arm, state, target, Some(src), ctx_ref)?;
        Ok(BranchResult {
            state: st,
            values: v,
        })
    };
    let ta = |rb: &mut RegionBuilder| build(rb, true_target);
    let fa = |rb: &mut RegionBuilder| build(rb, false_target);
    let gamma = lowerer.rb.gamma(cond, state, &vals, ta, fa)?;

    let total = ctx.entry_phis.len() + 1;
    let out: Vec<ValueId> = (0..total as u16).map(|i| gamma.result(i)).collect();
    Ok((gamma.state, out))
}

/// Entry leaf: control enters the loop at `entry_e` from `src`. Yields the
/// q index of that entry, then for each entry phi its incoming along this
/// entry arc (for phis at `entry_e`) or zero (for other entries' phis,
/// whose slots are written by a repetition arc before first use).
fn make_entry_leaf<'m>(
    lowerer: &mut RegionLowerer<'_, '_, 'm>,
    entry_e: BasicBlockId,
    src: BasicBlockId,
    ctx: &EntryCtx<'_, 'm>,
) -> color_eyre::Result<Vec<ValueId>> {
    let q_idx = ctx.entries.iter().position(|&e| e == entry_e).unwrap() as i32;
    let mut vals: Vec<ValueId> = Vec::with_capacity(ctx.entry_phis.len() + 1);
    vals.push(lowerer.rb.const_i32(q_idx));
    for ep in ctx.entry_phis {
        if ep.entry == entry_e {
            let (op, _) = phi_incoming_from(ep.phi, lowerer.fn_ctx.bb_mapper, |id| id == src)
                .ok_or_else(|| {
                    eyre!(
                        "entry phi {:?} has no incoming for entry arc from block {}",
                        ep.phi.dest,
                        src.0
                    )
                })?;
            vals.push(lowerer.operand(op)?);
        } else {
            let ty = lowerer
                .rb
                .graph
                .types
                .convert_type_ref(&ep.phi.to_type, lowerer.fn_ctx.llvm_mod)?;
            vals.push(lowerer.zero_of(ty)?);
        }
    }
    Ok(vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_parser::region::test_fixture::{TestFn, local_name, scc_for};
    use pretty_assertions::assert_eq;

    fn analyze<'a>(
        lowerer: &mut RegionLowerer<'_, '_, 'a>,
        topo: &crate::llvm_parser::region::test_fixture::SccTopology,
    ) -> color_eyre::Result<LoopLowerCtx<'a>> {
        lowerer.analyze_loop(topo.id, topo.header, &[topo.exit_arc])
    }

    #[test]
    fn analyze_loop_while_shape_reports_phi_and_loop_closed_classification() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f() {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %back ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 10
  br i1 %done, label %exit, label %back
back:
  br label %loop
exit:
  %out = phi i32 [ %next, %loop ]
  ret i32 %out
}
"#,
        );
        let topo = scc_for(&test_fn, "loop");
        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let ctx = analyze(lowerer, &topo).expect("analyze");
            assert_eq!(ctx.header_phis.len(), 1);
            assert_eq!(ctx.header_phis[0].phi.dest, local_name("i"));
            assert!(matches!(
                ctx.header_phis[0].next_operand,
                Operand::LocalOperand { name, .. } if name == &local_name("next")
            ));
            assert!(ctx.live_in_names.is_empty(), "no live-ins expected");
            assert_eq!(ctx.lcssa_bindings_per_exit.len(), 1);
            assert_eq!(ctx.lcssa_bindings_per_exit[0].len(), 1);
            let (lcssa_dest, binding) = &ctx.lcssa_bindings_per_exit[0][0];
            assert_eq!(*lcssa_dest, local_name("out"));
            assert!(matches!(binding, LcssaBinding::HeaderPhi { index: 0 }));
        });
    }

    #[test]
    fn analyze_loop_do_while_shape_reports_back_edge_lcssa() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f() {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %back ]
  %next = add i32 %i, 1
  br label %test
test:
  %done = icmp eq i32 %next, 10
  br i1 %done, label %exit, label %back
back:
  br label %loop
exit:
  %out = phi i32 [ %next, %test ]
  ret i32 %out
}
"#,
        );
        let topo = scc_for(&test_fn, "loop");
        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let ctx = analyze(lowerer, &topo).expect("analyze");
            assert_eq!(ctx.header_phis.len(), 1);
            assert_eq!(ctx.lcssa_bindings_per_exit.len(), 1);
            assert_eq!(ctx.lcssa_bindings_per_exit[0].len(), 1);
            assert!(matches!(
                ctx.lcssa_bindings_per_exit[0][0].1,
                LcssaBinding::HeaderPhi { index: 0 }
            ));
        });
    }

    #[test]
    fn analyze_loop_body_internal_loop_closed_value_allocates_extra_slot() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f() {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %back ]
  %doubled = shl i32 %i, 1
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 10
  br i1 %done, label %exit, label %back
back:
  br label %loop
exit:
  %out = phi i32 [ %doubled, %loop ]
  ret i32 %out
}
"#,
        );
        let topo = scc_for(&test_fn, "loop");
        test_fn.with_lowerer(FxHashMap::default(), |lowerer| {
            let ctx = analyze(lowerer, &topo).expect("analyze");
            assert_eq!(ctx.lcssa_extras.len(), 1);
            assert_eq!(ctx.lcssa_extras[0].name, local_name("doubled"));
            assert!(matches!(
                ctx.lcssa_bindings_per_exit[0][0].1,
                LcssaBinding::Extra { index: 0 }
            ));
            assert_eq!(ctx.inits.all().len(), 2);
            assert_eq!(ctx.inits.header_phis().len(), 1);
            assert_eq!(ctx.inits.live_ins().len(), 0);
            assert_eq!(ctx.inits.lcssa_extras().len(), 1);
        });
    }

    #[test]
    fn analyze_loop_does_not_double_count_phi_init_as_live_in() {
        let test_fn = TestFn::from_ir(
            r#"
define i32 @f(i32 %arg) {
entry:
  br label %loop
loop:
  %i = phi i32 [ %arg, %entry ], [ %next, %back ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 10
  br i1 %done, label %exit, label %back
back:
  br label %loop
exit:
  %out = phi i32 [ %next, %loop ]
  ret i32 %out
}
"#,
        );
        let topo = scc_for(&test_fn, "loop");
        let mut outer = FxHashMap::default();
        outer.insert(local_name("arg"), ValueId(100));
        test_fn.with_lowerer(outer, |lowerer| {
            let ctx = analyze(lowerer, &topo).expect("analyze");
            assert!(
                ctx.live_in_names.is_empty(),
                "phi init %arg should not appear as a live-in (got {:?})",
                ctx.live_in_names
            );
            assert_eq!(ctx.header_phis.len(), 1);
            assert_eq!(
                ctx.header_phis[0].init,
                ValueId(100),
                "header phi init should resolve %arg to its outer ValueId"
            );
        });
    }
}
