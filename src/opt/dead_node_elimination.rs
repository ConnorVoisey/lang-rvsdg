use color_eyre::eyre::eyre;

use crate::rvsdg::{
    MatchArmPool, MatchArmSpan, RVSDGMod, RegionId, RegionPool, RegionsSpan, State, U32Pool,
    U32Span, ValueId, ValueKind, ValuePool, ValuesSpan, function_graph::FunctionGraph,
};

/// Counters only this pass can produce cheaply: slot-level interface
/// shrink and adjacency pinning, each a single increment inside a loop
/// the pass runs anyway. Whole-graph deltas (values, regions, pools)
/// are measured by the pipeline driver around the pass instead
/// (see [`super::PassReport`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct DneEffects {
    /// Gamma input slots dropped because no arm reads the parameter.
    pub gamma_input_slots_dropped: u64,
    /// Theta loop variable slots dropped (all four faces dead).
    pub theta_loop_var_slots_dropped: u64,
    /// Result entries dropped from subregion result spans, counted per
    /// region: a gamma result slot dead across N arms counts N.
    pub result_entries_dropped: u64,
    /// Dead projections of live signature-fixed nodes (calls, loads,
    /// ...) kept alive for the projection adjacency contract; what a
    /// per-site slot representation could additionally remove.
    pub pinned_projections: u64,
}

impl RVSDGMod {
    pub fn opt_dead_node_elimination(&mut self) -> color_eyre::Result<DneEffects> {
        let mut effects = DneEffects::default();
        // Graph-sized mark scratch, reused across the function loop.
        let mut alive: Vec<bool> = Vec::new();
        for graph in self.graphs.iter_mut().flatten() {
            alive.clear();
            alive.resize(graph.value_kinds.len(), false);
            graph.mark_alive_nodes(&mut alive)?;
            graph.pin_projections(&mut alive, &mut effects);
            graph.remove_dead_nodes(&alive, &mut effects)?;
        }
        Ok(effects)
    }
}

impl FunctionGraph {
    /// Signature-fixed nodes (calls, loads, compare-and-swap, ...)
    /// own their projection layout: a live node keeps every
    /// projection, used or not, so the adjacency contract
    /// (projection_of) survives compaction. Gamma/theta
    /// projections are per-slot and stay dead with their slot. A
    /// projection's only operand is its node, so this sweep never
    /// creates new work for the mark loop.
    fn pin_projections(&self, alive: &mut [bool], effects: &mut DneEffects) {
        for index in 0..self.value_kinds.len() {
            if let ValueKind::Project { call, .. } = self.value_kinds[index] {
                let slotted = matches!(
                    self.get_value_kind(call),
                    ValueKind::Gamma { .. } | ValueKind::Theta { .. }
                );
                if !slotted && alive[call.0 as usize] && !alive[index] {
                    alive[index] = true;
                    effects.pinned_projections += 1;
                }
            }
        }
    }

    fn mark_alive_nodes(&self, alive: &mut [bool]) -> color_eyre::Result<()> {
        // Region 0 is the function body by the graph constructor's
        // convention. It is unique in this search: everything it
        // returns is assumed needed.
        let region = self.get_region(RegionId(0));

        let mut stack = self.value_pool.get(region.results).to_vec();
        stack.push(region.exit_state.0);
        stack.extend_from_slice(&region.params);
        stack.push(region.entry_state.0);

        while let Some(value_id) = stack.pop() {
            if alive[value_id.0 as usize] {
                continue;
            }
            alive[value_id.0 as usize] = true;

            // require a custom walker to traverse operands,
            // need state and to visit regions, theta and gamma must be handled specially to remove
            // redundant pass through
            match self.get_value_kind(value_id) {
                ValueKind::Fence { state, .. } => {
                    stack.push(state.0);
                }
                ValueKind::Load {
                    state, addr: val, ..
                }
                | ValueKind::Alloca {
                    state, count: val, ..
                }
                | ValueKind::AtomicLoad {
                    state, addr: val, ..
                } => {
                    stack.push(state.0);
                    stack.push(*val);
                }
                ValueKind::AtomicStore {
                    state, addr, value, ..
                }
                | ValueKind::Store {
                    state, addr, value, ..
                }
                | ValueKind::AtomicReadModifyWrite {
                    state, addr, value, ..
                } => {
                    stack.push(state.0);
                    stack.push(*addr);
                    stack.push(*value);
                }
                ValueKind::CompareAndSwap {
                    state,
                    addr,
                    expected,
                    desired,
                    ..
                } => {
                    stack.push(state.0);
                    stack.push(*addr);
                    stack.push(*expected);
                    stack.push(*desired);
                }
                ValueKind::Intrinsic { state, args, .. } | ValueKind::Call { state, args, .. } => {
                    stack.push(state.0);
                    stack.extend_from_slice(self.value_pool.get(*args));
                }
                ValueKind::CallIndirect {
                    state,
                    callee,
                    args,
                    ..
                } => {
                    stack.push(state.0);
                    stack.push(*callee);
                    stack.extend_from_slice(self.value_pool.get(*args));
                }
                ValueKind::RegionResult { values, state } => {
                    stack.push(state.0);
                    stack.extend_from_slice(self.value_pool.get(*values));
                }
                // A live theta always needs its repetition predicate and
                // its body's state chain. Its loop variable slots are NOT
                // blanket-marked: each slot is demanded individually,
                // through its projection (used after the loop) or its
                // body parameter (used by something live inside).
                ValueKind::Theta {
                    condition,
                    state,
                    region_id,
                    ..
                } => {
                    stack.push(*condition);
                    stack.push(state.0);
                    stack.push(self.regions[region_id.0 as usize].exit_state.0);
                }
                // Same for a live gamma: predicate, state, and every
                // arm's state chain. Inputs are demanded through arm
                // parameters, result slots through projections.
                ValueKind::Gamma {
                    condition,
                    state,
                    regions,
                    ..
                } => {
                    stack.push(*condition);
                    stack.push(state.0);
                    for &arm in self.region_pool.get(*regions) {
                        stack.push(self.regions[arm.0 as usize].exit_state.0);
                    }
                }
                // A used projection is a demanded result slot: the node
                // itself, plus the values feeding that slot inside each
                // subregion. Signature-fixed nodes (calls, loads, ...)
                // have no per-slot choice; their layout is pinned by the
                // sweep in opt_dead_node_elimination instead.
                ValueKind::Project { call, index } => {
                    stack.push(*call);
                    let index = *index;
                    match self.get_value_kind(*call) {
                        ValueKind::Gamma { regions, .. } => {
                            for &arm in self.region_pool.get(*regions) {
                                let arm_results = self.regions[arm.0 as usize].results;
                                stack.push(self.value_pool.get(arm_results)[index as usize]);
                            }
                        }
                        ValueKind::Theta { .. } => {
                            self.push_theta_slot_faces(*call, index, &mut stack);
                        }
                        _ => {}
                    }
                }
                // A used parameter is a demanded input slot of the
                // owning construct. The body region's parameters (region
                // 0, owner-less by the root convention) are the
                // function's own parameters, fed by callers rather than
                // graph edges; nothing to demand for those.
                ValueKind::RegionParam { index, region, .. } => {
                    if region.0 != 0 {
                        let index = *index;
                        let owner = self.regions[region.0 as usize].owner;
                        match self.get_value_kind(owner) {
                            // The outer value feeding the slot, and the slot's
                            // parameter in every sibling arm: slots are
                            // aligned across arms, so a slot read anywhere
                            // survives everywhere.
                            ValueKind::Gamma {
                                inputs, regions, ..
                            } => {
                                stack.push(self.value_pool.get(*inputs)[index as usize]);
                                for &arm in self.region_pool.get(*regions) {
                                    stack.push(self.regions[arm.0 as usize].params[index as usize]);
                                }
                            }
                            ValueKind::Theta { .. } => {
                                self.push_theta_slot_faces(owner, index as u16, &mut stack);
                            }
                            t => Err(eyre!(
                                "region {region:?} owned by non-construct value {owner:?} ({t:?})"
                            ))?,
                        }
                    }
                }
                // this is inefficent since we're rematching, but it prevents duping the logic of
                // the non special cases
                _ => self.for_each_value_operand(value_id, |op_id| stack.push(op_id)),
            }
        }

        Ok(())
    }

    /// A theta loop variable slot is one recurrence with four faces:
    /// the initial value, the body parameter, the body result, and the
    /// output projection. Any face demanded demands all four, so
    /// reconstruction sees slots as all-live or all-dead.
    fn push_theta_slot_faces(&self, theta: ValueId, index: u16, stack: &mut Vec<ValueId>) {
        let ValueKind::Theta {
            loop_vars,
            region_id,
            ..
        } = self.get_value_kind(theta)
        else {
            unreachable!("theta slot faces requested for non-theta value");
        };
        let body = &self.regions[region_id.0 as usize];
        stack.push(self.value_pool.get(*loop_vars)[index as usize]);
        stack.push(body.params[index as usize]);
        stack.push(self.value_pool.get(body.results)[index as usize]);
        stack.push(self.projection_of(theta, index));
    }

    #[tracing::instrument(skip_all)]
    fn remove_dead_nodes(
        &mut self,
        alive: &[bool],
        effects: &mut DneEffects,
    ) -> color_eyre::Result<()> {
        debug_assert_eq!(alive.len(), self.value_kinds.len());

        // Both mappers are plain prefix sums of the mark, complete
        // before any rewriting starts. Dead entries stay poisoned so a
        // remap through one panics instead of aliasing id 0.
        let mut value_mapper = vec![u32::MAX; self.value_kinds.len()];
        let mut live_values: u32 = 0;
        for (old, &is_alive) in alive.iter().enumerate() {
            if is_alive {
                value_mapper[old] = live_values;
                live_values += 1;
            }
        }
        // The body region (region 0, owner-less by the root convention)
        // always lives; every other region lives exactly as long as the
        // construct owning it.
        let mut region_mapper = vec![u32::MAX; self.regions.len()];
        let mut live_regions: u32 = 0;
        for (old, region) in self.regions.iter().enumerate() {
            if old == 0 || alive[region.owner.0 as usize] {
                region_mapper[old] = live_regions;
                live_regions += 1;
            }
        }

        // Slide live values down and rewrite their ids. Every reference
        // points backwards (construction is append-only), and targets
        // are always at or below their source, so processing old ids
        // ascending never clobbers an unread live value and every
        // mapper entry a value needs is already final. Spans are
        // repooled into fresh pools (each span is uniquely owned by the
        // field holding it, so it is visited exactly once); the old
        // pools, holes and all, are dropped when the new ones swap in.
        let mut fresh = FreshPools::default();
        let mut scratch: Vec<ValueId> = Vec::new();
        for old in 0..self.value_kinds.len() {
            if !alive[old] {
                continue;
            }
            let mut value = self.value_kinds[old].clone();
            remap_kind(
                &mut value,
                ValueId(old as u32),
                &mut RemapContext {
                    value_mapper: &value_mapper,
                    region_mapper: &region_mapper,
                    alive,
                    graph: self,
                    fresh: &mut fresh,
                    scratch: &mut scratch,
                    effects,
                },
            )?;
            self.value_kinds[value_mapper[old] as usize] = value;
            self.value_types[value_mapper[old] as usize] = self.value_types[old];
        }
        self.value_kinds.truncate(live_values as usize);
        self.value_types.truncate(live_values as usize);

        // Slide live regions the same way. This runs after the value
        // pass because remap_kind reads the OLD regions: parameter
        // lists drive the slot masks and index renumbering.
        for old in 0..self.regions.len() {
            let new = region_mapper[old];
            if new == u32::MAX {
                continue;
            }
            // Swap rather than clone: params/nodes are heap vecs, and
            // the evicted slot content is dead or already relocated,
            // never read again.
            self.regions.swap(new as usize, old);

            // Result slots of a gamma arm or theta body live and die
            // with their projection (old ids: projections sit directly
            // after their construct). The body region's results (region
            // 0, owner-less by the root convention) are the function's
            // ABI and are kept whole.
            let owner_old = self.regions[new as usize].owner;
            let keep_all_results = old == 0;
            let results = self.regions[new as usize].results;
            let new_results = repool_masked(
                &self.value_pool,
                &mut fresh.value_pool,
                &mut scratch,
                &value_mapper,
                results,
                |slot| keep_all_results || alive[owner_old.0 as usize + 1 + slot],
            );
            effects.result_entries_dropped += (results.len - new_results.len) as u64;

            let region = &mut self.regions[new as usize];
            region.results = new_results;
            if old != 0 {
                region.owner = ValueId(value_mapper[owner_old.0 as usize]);
            }
            // States are remapped poison-blind everywhere else by
            // design; these two are the only remaps with no assert
            // between them and a use-site panic, so check here.
            debug_assert!(
                alive[region.entry_state.0.0 as usize],
                "region {old}: entry state target is dead"
            );
            debug_assert!(
                alive[region.exit_state.0.0 as usize],
                "region {old}: exit state target is dead"
            );
            region.entry_state = State(ValueId(value_mapper[region.entry_state.0.0 as usize]));
            region.exit_state = State(ValueId(value_mapper[region.exit_state.0.0 as usize]));
            region.params.retain(|param| alive[param.0 as usize]);
            for param in region.params.iter_mut() {
                *param = ValueId(value_mapper[param.0 as usize]);
            }
            region.nodes.retain(|node| alive[node.0 as usize]);
            for node in region.nodes.iter_mut() {
                *node = ValueId(value_mapper[node.0 as usize]);
            }
        }
        self.regions.truncate(live_regions as usize);

        fresh.install(self);

        self.remap_interned_values(alive, &value_mapper);

        Ok(())
    }
}

/// Everything `remap_kind` needs, split by field so the borrows stay
/// disjoint: the OLD regions are read (parameter lists drive slot masks
/// and index renumbering) while the pools are written (span contents
/// rewritten in place).
/// The pools rebuilt by a compaction: live spans are repooled here and
/// the whole set swaps into the graph at the end, dropping the old
/// pools holes and all. Field names mirror the graph's.
#[derive(Default)]
struct FreshPools {
    value_pool: ValuePool,
    region_pool: RegionPool,
    u32_pool: U32Pool,
    match_arm_pool: MatchArmPool,
}

impl FreshPools {
    fn install(self, graph: &mut FunctionGraph) {
        graph.value_pool = self.value_pool;
        graph.region_pool = self.region_pool;
        graph.u32_pool = self.u32_pool;
        graph.match_arm_pool = self.match_arm_pool;
    }
}

/// Copy the slots `keep` selects from `span` into `new`, remapped
/// through `mapper` and staged in `scratch`. The single implementation
/// of the masked span copy, shared by the value pass (construct input
/// spans) and the region pass (result spans).
fn repool_masked(
    old: &ValuePool,
    new: &mut ValuePool,
    scratch: &mut Vec<ValueId>,
    mapper: &[u32],
    span: ValuesSpan,
    mut keep: impl FnMut(usize) -> bool,
) -> ValuesSpan {
    scratch.clear();
    for (slot, &value) in old.get(span).iter().enumerate() {
        if keep(slot) {
            scratch.push(ValueId(mapper[value.0 as usize]));
        }
    }
    new.push_slice(scratch)
}

struct RemapContext<'a> {
    value_mapper: &'a [u32],
    region_mapper: &'a [u32],
    alive: &'a [bool],
    /// Read side: the pre-compaction graph (regions and pools).
    graph: &'a FunctionGraph,
    /// Write side: the pools being rebuilt.
    fresh: &'a mut FreshPools,
    /// Reused staging buffer for masked span copies.
    scratch: &'a mut Vec<ValueId>,
    /// Slot-drop counters, incremented at the shrink sites.
    effects: &'a mut DneEffects,
}

impl RemapContext<'_> {
    fn map_value(&self, value: ValueId) -> ValueId {
        ValueId(self.value_mapper[value.0 as usize])
    }

    fn map_state(&self, state: State) -> State {
        State(self.map_value(state.0))
    }

    fn map_region(&self, region: RegionId) -> RegionId {
        RegionId(self.region_mapper[region.0 as usize])
    }

    /// Copy a span into the new pool with contents remapped, length
    /// unchanged (every entry of a live node's span is itself live).
    fn repool_values(&mut self, span: ValuesSpan) -> ValuesSpan {
        self.repool_values_masked(span, |_| true)
    }

    /// Copy the slots `keep` selects into the new pool, remapped.
    fn repool_values_masked(
        &mut self,
        span: ValuesSpan,
        keep: impl FnMut(usize) -> bool,
    ) -> ValuesSpan {
        repool_masked(
            &self.graph.value_pool,
            &mut self.fresh.value_pool,
            self.scratch,
            self.value_mapper,
            span,
            keep,
        )
    }

    /// Copy a regions span into the new pool, then remap the copy in
    /// place. Arms always survive with their construct, so the length
    /// is unchanged and no staging is needed.
    fn repool_regions(&mut self, span: RegionsSpan) -> RegionsSpan {
        let new_span = self
            .fresh
            .region_pool
            .push_slice(self.graph.region_pool.get(span));
        for region in self.fresh.region_pool.get_mut(new_span) {
            *region = RegionId(self.region_mapper[region.0 as usize]);
        }
        new_span
    }

    /// Copy a constant-u32 span (field index paths) verbatim.
    fn repool_u32(&mut self, span: U32Span) -> U32Span {
        self.fresh
            .u32_pool
            .push_slice(self.graph.u32_pool.get(span))
    }

    /// Copy a match-arm span (constant case-to-alternative pairs)
    /// verbatim.
    fn repool_match_arms(&mut self, span: MatchArmSpan) -> MatchArmSpan {
        self.fresh
            .match_arm_pool
            .push_slice(self.graph.match_arm_pool.get(span))
    }
}

/// Rewrite every id `kind` holds for the compacted graph: value and
/// region operands through the mappers, spans repooled into the fresh
/// pools (so dead spans and dropped tails are reclaimed when the pools
/// swap), slot indices renumbered over the live slots, and construct
/// input spans (gamma inputs, theta loop vars) shrunk to their live
/// slots. Exhaustive over `ValueKind`, so adding a variant forces a
/// decision here. `old_id` is the value's pre-compaction id (projection
/// adjacency is defined on old ids).
#[inline(always)]
fn remap_kind(
    kind: &mut ValueKind,
    old_id: ValueId,
    ctx: &mut RemapContext,
) -> color_eyre::Result<()> {
    match kind {
        ValueKind::Const(_)
        | ValueKind::ConstPoolRef(_)
        | ValueKind::GlobalRef(_)
        | ValueKind::FuncAddr(_) => {}
        ValueKind::Unary { operand, .. } => *operand = ctx.map_value(*operand),
        ValueKind::Cast { value, .. } | ValueKind::Freeze { value } => {
            *value = ctx.map_value(*value);
        }
        ValueKind::Binary { left, right, .. }
        | ValueKind::ICmp { left, right, .. }
        | ValueKind::FCmp { left, right, .. } => {
            *left = ctx.map_value(*left);
            *right = ctx.map_value(*right);
        }
        ValueKind::Ternary {
            condition,
            true_val,
            false_val,
        } => {
            *condition = ctx.map_value(*condition);
            *true_val = ctx.map_value(*true_val);
            *false_val = ctx.map_value(*false_val);
        }
        ValueKind::ExtractLane { vector, index } => {
            *vector = ctx.map_value(*vector);
            *index = ctx.map_value(*index);
        }
        ValueKind::InsertLane {
            vector,
            index,
            value,
        } => {
            *vector = ctx.map_value(*vector);
            *index = ctx.map_value(*index);
            *value = ctx.map_value(*value);
        }
        ValueKind::ShuffleLanes { left, right, mask } => {
            *left = ctx.map_value(*left);
            *right = ctx.map_value(*right);
            *mask = ctx.repool_values(*mask);
        }
        // Field index paths are constant u32 spans, not values;
        // repooled verbatim so dead nodes' spans are reclaimed.
        ValueKind::ExtractField { aggregate, indices } => {
            *aggregate = ctx.map_value(*aggregate);
            *indices = ctx.repool_u32(*indices);
        }
        ValueKind::InsertField {
            aggregate,
            value,
            indices,
        } => {
            *aggregate = ctx.map_value(*aggregate);
            *value = ctx.map_value(*value);
            *indices = ctx.repool_u32(*indices);
        }
        ValueKind::PtrOffset { base, indices, .. } => {
            *base = ctx.map_value(*base);
            *indices = ctx.repool_values(*indices);
        }
        ValueKind::Load { state, addr, .. } | ValueKind::AtomicLoad { state, addr, .. } => {
            *state = ctx.map_state(*state);
            *addr = ctx.map_value(*addr);
        }
        ValueKind::Store {
            state, addr, value, ..
        }
        | ValueKind::AtomicStore {
            state, addr, value, ..
        }
        | ValueKind::AtomicReadModifyWrite {
            state, addr, value, ..
        } => {
            *state = ctx.map_state(*state);
            *addr = ctx.map_value(*addr);
            *value = ctx.map_value(*value);
        }
        ValueKind::CompareAndSwap {
            state,
            addr,
            expected,
            desired,
            ..
        } => {
            *state = ctx.map_state(*state);
            *addr = ctx.map_value(*addr);
            *expected = ctx.map_value(*expected);
            *desired = ctx.map_value(*desired);
        }
        ValueKind::Alloca { state, count, .. } => {
            *state = ctx.map_state(*state);
            *count = ctx.map_value(*count);
        }
        ValueKind::Fence { state, .. } => *state = ctx.map_state(*state),
        // Match arms are constant case-to-alternative pairs, not
        // values; repooled verbatim.
        ValueKind::Match { input, arms, .. } => {
            *input = ctx.map_value(*input);
            *arms = ctx.repool_match_arms(*arms);
        }
        ValueKind::Intrinsic { state, args, .. } | ValueKind::Call { state, args, .. } => {
            *state = ctx.map_state(*state);
            *args = ctx.repool_values(*args);
        }
        ValueKind::CallIndirect {
            state,
            callee,
            args,
            ..
        } => {
            *state = ctx.map_state(*state);
            *callee = ctx.map_value(*callee);
            *args = ctx.repool_values(*args);
        }
        ValueKind::Theta {
            loop_vars,
            condition,
            state,
            region_id,
        } => {
            *condition = ctx.map_value(*condition);
            *state = ctx.map_state(*state);
            let alive = ctx.alive;
            let body = &ctx.graph.regions[region_id.0 as usize];
            // The slot mask here is the body parameter; the body
            // results shrink later against the projection. The mark
            // keeps a slot's four faces in lockstep -- hold it to that.
            for slot in 0..loop_vars.len as usize {
                debug_assert_eq!(
                    alive[body.params[slot].0 as usize],
                    alive[old_id.0 as usize + 1 + slot],
                    "theta {old_id:?} slot {slot}: parameter and projection liveness disagree"
                );
            }
            let body_params: &[ValueId] = &body.params;
            let repooled =
                ctx.repool_values_masked(*loop_vars, |slot| alive[body_params[slot].0 as usize]);
            ctx.effects.theta_loop_var_slots_dropped += (loop_vars.len - repooled.len) as u64;
            *loop_vars = repooled;
            *region_id = ctx.map_region(*region_id);
        }
        ValueKind::Gamma {
            condition,
            inputs,
            state,
            regions,
        } => {
            *condition = ctx.map_value(*condition);
            *state = ctx.map_state(*state);
            // Input slots are aligned across arms, so any arm's
            // parameter liveness is THE slot mask; arms themselves
            // always survive with their construct.
            let alive = ctx.alive;
            let arm0 = ctx.graph.region_pool.get(*regions)[0];
            let arm0_params: &[ValueId] = &ctx.graph.regions[arm0.0 as usize].params;
            let repooled =
                ctx.repool_values_masked(*inputs, |slot| alive[arm0_params[slot].0 as usize]);
            ctx.effects.gamma_input_slots_dropped += (inputs.len - repooled.len) as u64;
            *inputs = repooled;
            *regions = ctx.repool_regions(*regions);
        }
        // Projections are contiguous after their node, so the new index
        // is the count of live projections before this one. For
        // signature-fixed nodes every projection is live (the sweep in
        // opt_dead_node_elimination), so the count equals the old index
        // and nothing changes.
        ValueKind::Project { call, index } => {
            let first = call.0 as usize + 1;
            *index = ctx.alive[first..first + *index as usize]
                .iter()
                .filter(|live| **live)
                .count() as u16;
            *call = ctx.map_value(*call);
        }
        ValueKind::RegionParam { index, region, .. } => {
            // New index = live parameters before this one. The entry
            // state parameter sits one past the params list (its index
            // equals the list length), so counting the whole list gives
            // its new one-past position.
            let params = &ctx.graph.regions[region.0 as usize].params;
            *index = params[..*index as usize]
                .iter()
                .filter(|param| ctx.alive[param.0 as usize])
                .count() as u32;
            *region = ctx.map_region(*region);
        }
        ValueKind::RegionResult { .. } => {
            return Err(eyre!(
                "RegionResult {old_id:?} survived marking; it shares its span with its \
                 region's results, so remapping both would corrupt the pool"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::DneEffects;
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ConstValue, ICmpPred, Linkage, RVSDGMod, ValueId, ValueKind,
        builder::{BranchResult, LoopResult},
        func::FnResult,
        function_graph::FunctionGraph,
        types::{BOOL, I32, PtrType, TypeRef},
    };

    // Every doomed value in these tests is a Mul and every live value is
    // an Add, so assertions are just kind counts and survive the id
    // renumbering that rebuild-with-compaction performs.

    /// The single defined function graph these tests build. Ids captured
    /// during define_fn are ids of THIS graph.
    fn graph(m: &RVSDGMod) -> &FunctionGraph {
        let mut graphs = m.graphs.iter().flatten();
        let graph = graphs.next().expect("test defines one function");
        assert!(
            graphs.next().is_none(),
            "these tests assume a single function"
        );
        graph
    }

    fn graph_mut(m: &mut RVSDGMod) -> &mut FunctionGraph {
        m.graphs
            .iter_mut()
            .flatten()
            .next()
            .expect("test defines one function")
    }

    fn count_nodes(m: &RVSDGMod, pred: impl Fn(&ValueKind) -> bool) -> usize {
        let g = graph(m);
        g.regions
            .iter()
            .flat_map(|region| region.nodes.iter())
            .filter(|id| pred(g.get_value_kind(**id)))
            .count()
    }

    fn count_muls(m: &RVSDGMod) -> usize {
        count_nodes(m, |kind| {
            matches!(
                kind,
                ValueKind::Binary {
                    op: BinaryOp::Mul,
                    ..
                }
            )
        })
    }

    /// The single node matching `pred`, by id. Panics unless exactly one
    /// exists, so lookups cannot silently grab the wrong construct.
    fn single_node(m: &RVSDGMod, pred: impl Fn(&ValueKind) -> bool) -> ValueId {
        let g = graph(m);
        let mut found = None;
        for region in &g.regions {
            for &id in &region.nodes {
                if pred(g.get_value_kind(id)) {
                    assert!(found.is_none(), "expected exactly one matching node");
                    found = Some(id);
                }
            }
        }
        found.expect("expected exactly one matching node")
    }

    fn ptr_ty(m: &mut RVSDGMod) -> TypeRef {
        let id = m.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        });
        TypeRef::Ptr(id)
    }

    fn assert_verified(m: &RVSDGMod) {
        let errs = m.verify();
        assert!(errs.is_empty(), "graph failed verification: {errs:?}");
    }

    fn mark_alive(m: &RVSDGMod) -> Vec<bool> {
        let g = graph(m);
        let mut alive = vec![false; g.value_kinds.len()];
        g.mark_alive_nodes(&mut alive).unwrap();
        alive
    }

    // -- Mark phase --------------------------------------------------
    //
    // These inspect the alive vec directly, so a marking bug is caught
    // here as the exact value that was wrongly marked or missed, rather
    // than as a structural diff after removal.

    /// Roots are the function region's results; the walk reaches the
    /// returned value's operands and nothing else.
    #[test]
    fn mark_walks_operands_from_results() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        let ids: Cell<Option<(ValueId, ValueId, ValueId, ValueId)>> = Cell::new(None);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let dead = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, y, I32);
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            ids.set(Some((x, y, dead, live)));
            Ok(FnResult {
                state,
                values: vec![live],
            })
        })
        .unwrap();
        let (x, y, dead, live) = ids.get().unwrap();

        let alive = mark_alive(&m);

        assert!(alive[live.0 as usize], "returned value is marked");
        assert!(alive[x.0 as usize], "operand of a live value is marked");
        assert!(alive[y.0 as usize], "operand of a live value is marked");
        assert!(!alive[dead.0 as usize], "unreferenced value is not marked");
    }

    /// Nothing loads the slot back, so the alloca and both stores are
    /// reachable only through exit_state and then each node's state
    /// operand. The whole chain must be marked.
    #[test]
    fn mark_follows_state_chain_from_exit_state() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut m);
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        let ids: Cell<Option<(ValueId, ValueId, ValueId)>> = Cell::new(None);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr, None);
            let s1 = rb.store(alloc.state, alloc.ptr, x, None, false);
            let two = rb.const_i32(2);
            let s2 = rb.store(s1, alloc.ptr, two, None, false);
            ids.set(Some((alloc.state.0, s1.0, s2.0)));
            Ok(FnResult {
                state: s2,
                values: vec![x],
            })
        })
        .unwrap();
        let (alloca, store_1, store_2) = ids.get().unwrap();

        let alive = mark_alive(&m);

        assert!(alive[store_2.0 as usize], "exit state seeds the last store");
        assert!(
            alive[store_1.0 as usize],
            "state operand reaches the store one link back"
        );
        assert!(
            alive[alloca.0 as usize],
            "state operand reaches the alloca at the chain head"
        );
    }

    /// The self-feeding recurrence at mark level: variable `j` is only
    /// read to compute its own next value and never read after the
    /// loop, so no face of its slot may be marked. Demand-driven
    /// marking never enters the cycle; the `i` slot (drives the
    /// condition and the return value) is fully marked.
    #[test]
    fn mark_leaves_self_feeding_theta_slot_dead() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        let ids: Cell<Option<(ValueId, ValueId, ValueId, ValueId)>> = Cell::new(None);
        m.define_fn(f, |rb, state| {
            let init_i = rb.param(0);
            let init_j = rb.param(1);
            let res = rb.theta(state, &[init_i, init_j], |rb| {
                let i = rb.param(0);
                let j = rb.param(1);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), i, one, I32);
                let two = rb.const_i32(2);
                let next_j = rb.binary(BinaryOp::Mul, ArithFlags::default(), j, two, I32);
                let five = rb.const_i32(5);
                let condition = rb.icmp(ICmpPred::SignedLt, next_i, five);
                ids.set(Some((j, next_j, next_i, ValueId(0))));
                Ok(LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_i, next_j],
                })
            })?;
            let (j_param, next_j, next_i, _) = ids.get().unwrap();
            ids.set(Some((j_param, next_j, next_i, res.result(1))));
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();
        let (j_param, next_j, next_i, j_projection) = ids.get().unwrap();

        let alive = mark_alive(&m);

        assert!(!alive[j_param.0 as usize], "dead slot's body parameter");
        assert!(!alive[next_j.0 as usize], "dead slot's update computation");
        assert!(!alive[j_projection.0 as usize], "dead slot's projection");
        assert!(alive[next_i.0 as usize], "live slot's update computation");
    }

    /// A gamma input no arm ever reads: the outer value feeding it and
    /// the matching parameter in both arms stay unmarked, while the
    /// gamma itself (its result is returned) is fully live.
    #[test]
    fn mark_leaves_unused_gamma_input_slot_dead() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        let ids: Cell<Option<(ValueId, ValueId, ValueId)>> = Cell::new(None);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let dead_input = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, x, I32);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let true_param: Cell<Option<ValueId>> = Cell::new(None);
            let false_param: Cell<Option<ValueId>> = Cell::new(None);
            let res = rb.gamma(
                predicate,
                state,
                &[dead_input],
                |rb| {
                    true_param.set(Some(rb.param(0)));
                    let zero = rb.const_i32(0);
                    Ok(BranchResult {
                        state,
                        values: vec![zero],
                    })
                },
                |rb| {
                    false_param.set(Some(rb.param(0)));
                    let one = rb.const_i32(1);
                    Ok(BranchResult {
                        state,
                        values: vec![one],
                    })
                },
            )?;
            ids.set(Some((
                dead_input,
                true_param.get().unwrap(),
                false_param.get().unwrap(),
            )));
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();
        let (dead_input, true_param, false_param) = ids.get().unwrap();

        let alive = mark_alive(&m);

        assert!(
            !alive[dead_input.0 as usize],
            "outer value feeding an unread input slot"
        );
        assert!(!alive[true_param.0 as usize], "unread parameter, arm 0");
        assert!(!alive[false_param.0 as usize], "unread parameter, arm 1");
    }

    // -- Removal phase -----------------------------------------------
    //
    // These hand remove_dead_nodes a crafted alive slice, so removal is
    // exercised without depending on marking being correct.

    /// One value flagged dead: it leaves the region, everything else
    /// (nodes, params, results) survives rewiring intact.
    #[test]
    fn remove_drops_flagged_nodes_and_rewires() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let _dead = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, y, I32);
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            Ok(FnResult {
                state,
                values: vec![live],
            })
        })
        .unwrap();
        let nodes_before = count_nodes(&m, |_| true);
        // RegionResult values are never marked by the real mark phase
        // (remove_dead_nodes refuses them: they share their span with
        // the region's results), so a valid crafted slice excludes
        // them too.
        let alive: Vec<bool> = graph(&m)
            .value_kinds
            .iter()
            .map(|v| {
                !matches!(
                    v,
                    ValueKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    } | ValueKind::RegionResult { .. }
                )
            })
            .collect();

        graph_mut(&mut m)
            .remove_dead_nodes(&alive, &mut DneEffects::default())
            .unwrap();

        assert_verified(&m);
        assert_eq!(count_muls(&m), 0, "flagged node removed from its region");
        assert_eq!(
            count_nodes(&m, |_| true),
            nodes_before - 1,
            "exactly the flagged node is gone"
        );
        assert_eq!(
            graph(&m)
                .regions
                .iter()
                .map(|r| r.params.len())
                .sum::<usize>(),
            2,
            "region parameters survive"
        );
    }

    /// Everything alive: removal must be an exact no-op on structure.
    #[test]
    fn remove_with_all_alive_changes_nothing() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let doubled = rb.binary(BinaryOp::Add, ArithFlags::default(), x, x, I32);
            Ok(FnResult {
                state,
                values: vec![doubled],
            })
        })
        .unwrap();
        let nodes_before = count_nodes(&m, |_| true);
        let alive: Vec<bool> = graph(&m)
            .value_kinds
            .iter()
            .map(|v| !matches!(v, ValueKind::RegionResult { .. }))
            .collect();

        graph_mut(&mut m)
            .remove_dead_nodes(&alive, &mut DneEffects::default())
            .unwrap();

        assert_verified(&m);
        assert_eq!(count_nodes(&m, |_| true), nodes_before);
    }

    /// A dead gamma leaves dead spans behind (its inputs, both arms'
    /// results, its regions span): reconstruction repools, so the pools
    /// shrink to live spans only instead of keeping holes.
    #[test]
    fn pools_are_compacted() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let _unused = rb.gamma(
                predicate,
                state,
                &[x, y],
                |rb| {
                    let a = rb.param(0);
                    Ok(BranchResult {
                        state,
                        values: vec![a],
                    })
                },
                |rb| {
                    let b = rb.param(1);
                    Ok(BranchResult {
                        state,
                        values: vec![b],
                    })
                },
            )?;
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            Ok(FnResult {
                state,
                values: vec![live],
            })
        })
        .unwrap();
        let value_pool_before = graph(&m).value_pool.len();
        let region_pool_before = graph(&m).region_pool.len();

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert!(
            graph(&m).value_pool.len() < value_pool_before,
            "dead gamma spans reclaimed: {} -> {}",
            value_pool_before,
            graph(&m).value_pool.len()
        );
        assert!(
            graph(&m).region_pool.len() < region_pool_before,
            "dead regions span reclaimed: {} -> {}",
            region_pool_before,
            graph(&m).region_pool.len()
        );
        // Only the function region's results span (one entry) remains
        // in the value pool.
        assert_eq!(graph(&m).value_pool.len(), 1);
        assert_eq!(graph(&m).region_pool.len(), 0);
    }

    /// Layer-1 observability: the pipeline returns one report per pass
    /// carrying the driver-measured shape delta and the pass's own slot
    /// counters, with exact values for a known graph.
    #[test]
    fn pass_report_carries_shape_delta_and_slot_counters() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            // One dead gamma input slot: neither arm reads param 0.
            let dead_input = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, x, I32);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let picked = rb.gamma(
                predicate,
                state,
                &[dead_input],
                |rb| {
                    let zero = rb.const_i32(0);
                    Ok(BranchResult {
                        state,
                        values: vec![zero],
                    })
                },
                |rb| {
                    let one = rb.const_i32(1);
                    Ok(BranchResult {
                        state,
                        values: vec![one],
                    })
                },
            )?;
            // One dead theta slot: j feeds only its own next value. The
            // gamma output inits the LIVE slot i, so the gamma's result
            // slot stays live and the counters below stay independent.
            let res = rb.theta(picked.state, &[picked.result(0), x], |rb| {
                let i = rb.param(0);
                let j = rb.param(1);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), i, one, I32);
                let next_j = rb.binary(BinaryOp::Mul, ArithFlags::default(), j, one, I32);
                let five = rb.const_i32(5);
                let condition = rb.icmp(ICmpPred::SignedLt, next_i, five);
                Ok(LoopResult {
                    condition,
                    next_state: picked.state,
                    next_vars: vec![next_i, next_j],
                })
            })?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();

        let pipeline = m.optimise_default(true).unwrap();

        assert_eq!(pipeline.passes.len(), 1);
        assert!(
            pipeline.pre_verify_duration > std::time::Duration::ZERO,
            "verify_all times the pre-pass verification"
        );
        let report = &pipeline.passes[0];
        assert_eq!(report.pass, "DeadNodeElimination");
        assert!(
            report.verify_duration > std::time::Duration::ZERO,
            "verify_all times each post-pass verification"
        );
        assert!(
            report.shape_after.values < report.shape_before.values,
            "shape delta shows removal: {} -> {}",
            report.shape_before.values,
            report.shape_after.values
        );
        assert!(report.shape_after.value_pool_entries < report.shape_before.value_pool_entries);
        let crate::opt::PassEffects::DeadNodeElimination(effects) = report.effects;
        assert_eq!(effects.gamma_input_slots_dropped, 1);
        assert_eq!(effects.theta_loop_var_slots_dropped, 1);
        // Exactly the dead theta slot's body result entry: the gamma's
        // result slot is live (it inits the live theta slot), so both
        // arms keep their entries.
        assert_eq!(effects.result_entries_dropped, 1);
    }

    // -- Basic cases -------------------------------------------------

    /// An unused pure computation alongside a used one: the unused Mul
    /// must go, the returned Add must stay.
    #[test]
    fn dead_pure_node_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let _dead = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, y, I32);
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            Ok(FnResult {
                state,
                values: vec![live],
            })
        })
        .unwrap();
        assert_eq!(count_muls(&m), 1);

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(count_muls(&m), 0, "unused Mul should be removed");
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Binary { .. })),
            1,
            "the returned Add must survive"
        );
    }

    /// A dead value whose operand is only used by other dead values: the
    /// whole chain must go, not just the last link.
    #[test]
    fn dead_chain_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let dead_a = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, x, I32);
            let _dead_b = rb.binary(BinaryOp::Mul, ArithFlags::default(), dead_a, x, I32);
            Ok(FnResult {
                state,
                values: vec![x],
            })
        })
        .unwrap();
        assert_eq!(count_muls(&m), 2);

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(count_muls(&m), 0, "whole dead chain should be removed");
    }

    /// A function where everything is reachable from the results: the
    /// pass must be a no-op on node membership.
    #[test]
    fn fully_live_function_is_untouched() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            let scaled = rb.binary(BinaryOp::Mul, ArithFlags::default(), sum, y, I32);
            Ok(FnResult {
                state,
                values: vec![scaled],
            })
        })
        .unwrap();
        let nodes_before = count_nodes(&m, |_| true);

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(
            count_nodes(&m, |_| true),
            nodes_before,
            "a fully live function must keep every node"
        );
    }

    /// Stores are reachable ONLY through state edges (nothing loads the
    /// slot back). Marking that ignores state operands would delete the
    /// entire chain; every side effect on the live chain must survive.
    #[test]
    fn state_chain_is_kept_alive() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut m);
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr, None);
            let s1 = rb.store(alloc.state, alloc.ptr, x, None, false);
            let two = rb.const_i32(2);
            let s2 = rb.store(s1, alloc.ptr, two, None, false);
            Ok(FnResult {
                state: s2,
                values: vec![x],
            })
        })
        .unwrap();

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Store { .. })),
            2,
            "stores on the live state chain must survive"
        );
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Alloca { .. })),
            1,
            "the alloca the stores write through must survive"
        );
    }

    /// A dead pure value sitting between two live stores: the stores
    /// stay, the value between them goes.
    #[test]
    fn dead_value_between_state_ops_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut m);
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr, None);
            let s1 = rb.store(alloc.state, alloc.ptr, x, None, false);
            let _dead = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, x, I32);
            let s2 = rb.store(s1, alloc.ptr, x, None, false);
            Ok(FnResult {
                state: s2,
                values: vec![x],
            })
        })
        .unwrap();

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(count_muls(&m), 0);
        assert_eq!(count_nodes(&m, |k| matches!(k, ValueKind::Store { .. })), 2);
    }

    /// A gamma whose outputs are unused and whose state output is
    /// bypassed (the function threads its entry state to the result):
    /// the whole construct, its projections, and its arm contents die.
    #[test]
    fn dead_gamma_construct_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let _unused = rb.gamma(
                predicate,
                state,
                &[x, y],
                |rb| {
                    let a = rb.param(0);
                    let b = rb.param(1);
                    let v = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![v],
                    })
                },
                |rb| {
                    let a = rb.param(0);
                    let b = rb.param(1);
                    let v = rb.binary(BinaryOp::Mul, ArithFlags::default(), b, a, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![v],
                    })
                },
            )?;
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            Ok(FnResult {
                state,
                values: vec![live],
            })
        })
        .unwrap();

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Gamma { .. })),
            0,
            "an unused pure gamma must be removed entirely"
        );
        assert_eq!(count_muls(&m), 0, "arm contents die with the construct");
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Project { .. })),
            0,
            "projections of a dead construct must not survive it"
        );
    }

    /// A live gamma with a dead computation inside one arm: the construct
    /// and its used results stay, the arm-local dead value goes.
    #[test]
    fn dead_node_inside_gamma_arm_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let res = rb.gamma(
                predicate,
                state,
                &[x, y],
                |rb| {
                    let a = rb.param(0);
                    let b = rb.param(1);
                    let _dead = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                    let v = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![v],
                    })
                },
                |rb| {
                    let a = rb.param(0);
                    let v = rb.binary(BinaryOp::Add, ArithFlags::default(), a, a, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![v],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();
        assert_eq!(count_muls(&m), 1);

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        assert_eq!(count_muls(&m), 0, "dead value inside a live arm removed");
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Gamma { .. })),
            1,
            "the live gamma itself survives"
        );
    }

    // -- Dead slot cases ---------------------------------------------

    /// A gamma with two result slots where only slot 0 is consumed: the
    /// unused slot must be removed from every arm's results, its
    /// projection must go, and the arm computations that only fed it
    /// become dead.
    #[test]
    fn unused_gamma_result_slot_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let res = rb.gamma(
                predicate,
                state,
                &[x, y],
                |rb| {
                    let a = rb.param(0);
                    let b = rb.param(1);
                    let used = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
                    let unused = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![used, unused],
                    })
                },
                |rb| {
                    let a = rb.param(0);
                    let b = rb.param(1);
                    let used = rb.binary(BinaryOp::Add, ArithFlags::default(), b, a, I32);
                    let unused = rb.binary(BinaryOp::Mul, ArithFlags::default(), b, a, I32);
                    Ok(BranchResult {
                        state,
                        values: vec![used, unused],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();
        assert_eq!(count_muls(&m), 2);

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        let gamma = single_node(&m, |k| matches!(k, ValueKind::Gamma { .. }));
        let g = graph(&m);
        let ValueKind::Gamma { regions, .. } = *g.get_value_kind(gamma) else {
            unreachable!();
        };
        for &arm in g.region_pool.get(regions) {
            assert_eq!(
                g.value_pool.get(g.regions[arm.0 as usize].results).len(),
                1,
                "unused result slot removed from every arm"
            );
        }
        assert_eq!(count_muls(&m), 0, "values feeding only the dead slot die");
        assert_eq!(
            count_nodes(
                &m,
                |k| matches!(k, ValueKind::Project { call, .. } if *call == gamma)
            ),
            1,
            "only the consumed projection remains"
        );
    }

    /// A gamma input never referenced by any arm: the input slot and the
    /// matching arm parameter must go, and the outer value that only fed
    /// that slot becomes dead.
    #[test]
    fn unused_gamma_input_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let dead_input = rb.binary(BinaryOp::Mul, ArithFlags::default(), x, x, I32);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let res = rb.gamma(
                predicate,
                state,
                &[dead_input],
                |rb| {
                    let zero = rb.const_i32(0);
                    Ok(BranchResult {
                        state,
                        values: vec![zero],
                    })
                },
                |rb| {
                    let one = rb.const_i32(1);
                    Ok(BranchResult {
                        state,
                        values: vec![one],
                    })
                },
            )?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        let gamma = single_node(&m, |k| matches!(k, ValueKind::Gamma { .. }));
        let g = graph(&m);
        let ValueKind::Gamma {
            inputs, regions, ..
        } = g.get_value_kind(gamma)
        else {
            unreachable!();
        };
        assert_eq!(
            g.value_pool.get(*inputs).len(),
            0,
            "unreferenced input slot removed"
        );
        for &arm in g.region_pool.get(*regions) {
            assert_eq!(
                g.regions[arm.0 as usize].params.len(),
                0,
                "matching arm parameter removed"
            );
        }
        assert_eq!(
            count_muls(&m),
            0,
            "the outer value that only fed the dead slot dies with it"
        );
    }

    /// A theta loop variable that is passed through unchanged, unused by
    /// the body and unused after the loop: its slot must be removed from
    /// the loop vars, the body parameters, and the body results.
    #[test]
    fn unused_theta_loop_var_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let init_i = rb.param(0);
            let unused_init = rb.param(1);
            let res = rb.theta(state, &[init_i, unused_init], |rb| {
                let i = rb.param(0);
                let unused = rb.param(1);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), i, one, I32);
                let five = rb.const_i32(5);
                let condition = rb.icmp(ICmpPred::SignedLt, next_i, five);
                Ok(LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_i, unused],
                })
            })?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        let theta = single_node(&m, |k| matches!(k, ValueKind::Theta { .. }));
        let g = graph(&m);
        let ValueKind::Theta {
            loop_vars,
            region_id,
            ..
        } = *g.get_value_kind(theta)
        else {
            unreachable!();
        };
        assert_eq!(
            g.value_pool.get(loop_vars).len(),
            1,
            "pass-through loop variable removed from the theta inputs"
        );
        let body = &g.regions[region_id.0 as usize];
        assert_eq!(body.params.len(), 1, "matching body parameter removed");
        assert_eq!(
            g.value_pool.get(body.results).len(),
            1,
            "matching body result slot removed"
        );
    }

    /// The self-feeding cycle: a loop variable that IS used in the body,
    /// but only to compute its own next value, and never read after the
    /// loop. Naive operand-following marks it alive through the cycle;
    /// slot-level liveness must see that no live sink ever reads it.
    #[test]
    fn self_referential_dead_loop_var_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let init_i = rb.param(0);
            let init_j = rb.param(1);
            let res = rb.theta(state, &[init_i, init_j], |rb| {
                let i = rb.param(0);
                let j = rb.param(1);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), i, one, I32);
                let two = rb.const_i32(2);
                let next_j = rb.binary(BinaryOp::Mul, ArithFlags::default(), j, two, I32);
                let five = rb.const_i32(5);
                let condition = rb.icmp(ICmpPred::SignedLt, next_i, five);
                Ok(LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_i, next_j],
                })
            })?;
            Ok(FnResult {
                state: res.state,
                values: vec![res.result(0)],
            })
        })
        .unwrap();
        assert_eq!(count_muls(&m), 1);

        m.opt_dead_node_elimination().unwrap();

        assert_verified(&m);
        let theta = single_node(&m, |k| matches!(k, ValueKind::Theta { .. }));
        let g = graph(&m);
        let ValueKind::Theta { loop_vars, .. } = *g.get_value_kind(theta) else {
            unreachable!();
        };
        assert_eq!(
            g.value_pool.get(loop_vars).len(),
            1,
            "self-feeding but externally unread loop variable removed"
        );
        assert_eq!(count_muls(&m), 0, "its update computation dies with it");
    }
}
