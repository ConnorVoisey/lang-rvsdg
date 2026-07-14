//! Read-only census over a constructed RVSDG.
//!
//! Two jobs (see graph_stats_plan.md): design input for the optimizer
//! substrate (rebuild vs mutate, holes vs compaction, use-list strategy,
//! how far alias analysis must go) and, once passes exist, a
//! before/after shape diff that quantifies each pass and catches
//! hygiene regressions correctness tests cannot see.
//!
//! Several queries here are deliberate DRY RUNS of pass analyses -- the
//! duplicate-node map (common node elimination), the foldable count
//! (constant folding), the invariance test (loop-invariant motion), the
//! promotion candidate scan -- so the census and the eventual passes
//! share one implementation of each rule. Base-object resolution lives
//! in `rvsdg::alias` for the same reason.

use std::io;
use std::mem::size_of;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;

use crate::rvsdg::{
    ConstValue, RVSDGMod, RegionId, Value, ValueId, ValueKind,
    alias::{BaseObject, may_alias_resolved},
    func::{FnAttrs, ModRef},
    ops::IntrinsicOp,
    verify::scope::Owner,
};

/// Census of one function's region tree. Distributions keep their raw
/// samples; the CSV row derives its aggregates from them.
#[derive(Debug, Default, Clone)]
pub struct FunctionCensus {
    pub name: String,
    /// Values owned by the tree (region nodes + params).
    pub values: u64,
    pub regions: u64,
    pub region_node_counts: Vec<u32>,
    pub max_depth: u32,
    pub max_loop_depth: u32,
    pub gammas: u64,
    pub gamma_arities: Vec<u32>,
    pub thetas: u64,
    pub theta_arities: Vec<u32>,
    pub matches: u64,
    pub projections: u64,

    // Pass-through freight: construct output slots that merely carry an
    // unchanged value through (every gamma arm yields its own image of
    // the same input; a theta result reduces to its own parameter).
    // Each such slot is a Project node (plus params/results plumbing)
    // that a thread-only-what-is-used construction would not create.
    pub gamma_outputs: u64,
    pub gamma_passthrough: u64,
    pub theta_outputs: u64,
    pub theta_passthrough: u64,
    /// Gamma arm result entries, and how many of them are poison (the
    /// value only exists on another path): sizes a sparse-per-arm
    /// results design against the dense-slot one.
    pub gamma_result_entries: u64,
    pub gamma_poison_results: u64,

    // Memory and state census, over theta bodies (whole subtrees).
    pub theta_mem_ops: Vec<u32>,
    pub theta_distinct_bases: Vec<u32>,
    pub addr_external: u64,
    pub addr_computable: u64,
    pub addr_varying: u64,
    pub calls_in_thetas: u64,
    pub calls_readonly: u64,

    // Scalar promotion dry run (v1 rules), one verdict per store
    // directly in a theta body.
    pub promotion_candidates: u64,
    pub bail_call: u64,
    pub bail_alias: u64,
    pub bail_nested: u64,
    pub bail_sync: u64,
    pub bail_varying: u64,

    // Rewrite-opportunity dry runs.
    pub dup_pure_in_region: u64,
    pub dup_pure_cross_region: u64,
    pub foldable: u64,
    pub licm_movable: u64,
}

/// One per-function CSV row. Serde keeps the header and the values in
/// sync by construction; aggregates are derived from the census samples
/// in [`FunctionCensus::row`].
#[derive(Debug, Serialize)]
pub struct FunctionRow<'a> {
    pub module: &'a str,
    pub function: &'a str,
    pub values: u64,
    pub regions: u64,
    pub max_region_nodes: u32,
    pub p50_region_nodes: u32,
    pub max_depth: u32,
    pub max_loop_depth: u32,
    pub gammas: u64,
    pub thetas: u64,
    pub theta_arity_max: u32,
    pub matches: u64,
    pub projections: u64,
    pub gamma_outputs: u64,
    pub gamma_passthrough: u64,
    pub theta_outputs: u64,
    pub theta_passthrough: u64,
    pub gamma_result_entries: u64,
    pub gamma_poison_results: u64,
    pub theta_mem_ops_max: u32,
    pub theta_bases_p90: u32,
    pub addr_external: u64,
    pub addr_computable: u64,
    pub addr_varying: u64,
    pub candidates: u64,
    pub bail_call: u64,
    pub bail_alias: u64,
    pub bail_nested: u64,
    pub bail_sync: u64,
    pub bail_varying: u64,
    pub dup_pure_region: u64,
    pub dup_pure_cross: u64,
    pub foldable: u64,
    pub licm_movable: u64,
    pub calls_in_thetas: u64,
    pub calls_readonly: u64,
}

impl FunctionCensus {
    pub fn row<'a>(&'a self, module: &'a str) -> FunctionRow<'a> {
        FunctionRow {
            module,
            function: &self.name,
            values: self.values,
            regions: self.regions,
            max_region_nodes: self.region_node_counts.iter().copied().max().unwrap_or(0),
            p50_region_nodes: percentile(&self.region_node_counts, 50),
            max_depth: self.max_depth,
            max_loop_depth: self.max_loop_depth,
            gammas: self.gammas,
            thetas: self.thetas,
            theta_arity_max: self.theta_arities.iter().copied().max().unwrap_or(0),
            matches: self.matches,
            projections: self.projections,
            gamma_outputs: self.gamma_outputs,
            gamma_passthrough: self.gamma_passthrough,
            theta_outputs: self.theta_outputs,
            theta_passthrough: self.theta_passthrough,
            gamma_result_entries: self.gamma_result_entries,
            gamma_poison_results: self.gamma_poison_results,
            theta_mem_ops_max: self.theta_mem_ops.iter().copied().max().unwrap_or(0),
            theta_bases_p90: percentile(&self.theta_distinct_bases, 90),
            addr_external: self.addr_external,
            addr_computable: self.addr_computable,
            addr_varying: self.addr_varying,
            candidates: self.promotion_candidates,
            bail_call: self.bail_call,
            bail_alias: self.bail_alias,
            bail_nested: self.bail_nested,
            bail_sync: self.bail_sync,
            bail_varying: self.bail_varying,
            dup_pure_region: self.dup_pure_in_region,
            dup_pure_cross: self.dup_pure_cross_region,
            foldable: self.foldable,
            licm_movable: self.licm_movable,
            calls_in_thetas: self.calls_in_thetas,
            calls_readonly: self.calls_readonly,
        }
    }
}

/// Bytes held by each backing array of the graph, so representation
/// debates start from a measured budget instead of hand arithmetic.
/// Interner tables (types/signatures/constants) are reported by COUNT
/// elsewhere; their heap contents (strings, nested vecs) make byte
/// figures dishonest without deep traversal.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct MemoryBudget {
    pub values_bytes: usize,
    pub value_pool_bytes: usize,
    pub region_structs_bytes: usize,
    /// Heap entries of every region's params/nodes Vec.
    pub region_lists_bytes: usize,
    pub region_pool_bytes: usize,
    pub u32_pool_bytes: usize,
    pub match_arm_pool_bytes: usize,
}

/// Where the value_pool's entries come from, span kind by span kind.
/// Prices representation changes: implicit outputs shrink some of
/// these and widen others.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct SpanComposition {
    pub gamma_inputs: usize,
    /// Every region's results span (function bodies, gamma arms, theta
    /// bodies alike; RegionResult values reuse the same span).
    pub region_results: usize,
    pub theta_loop_vars: usize,
    /// Call, CallIndirect and Intrinsic argument spans.
    pub call_args: usize,
    pub ptr_offset_indices: usize,
    pub shuffle_masks: usize,
    /// pool length minus everything above (replaced or orphaned spans).
    pub unaccounted: usize,
}

/// Wall-clock per pipeline phase, filled by the DRIVER (the census
/// library stays measurement-free so it can never contaminate compile
/// profiles).
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct PhaseTiming {
    pub frontend_and_parse_ms: u64,
    pub construction_ms: u64,
    pub verify_ms: u64,
    pub census_ms: u64,
}

/// Census of one module.
#[derive(Debug, Default)]
pub struct ModuleCensus {
    pub mod_name: String,
    pub total_values: u64,
    pub live_values: u64,
    pub kind_counts: FxHashMap<&'static str, u64>,
    /// Projection counts grouped by the kind of the projected node.
    pub projections_by_parent: FxHashMap<&'static str, u64>,
    /// Value-operand fan-out per value (index = value id).
    pub fanout: Vec<u32>,
    pub top_fanout: Vec<(ValueId, &'static str, u32)>,
    /// Project values with zero uses: output slots exported by a
    /// construct that nothing consumes (dead-node elimination fodder).
    pub dead_projections: u64,
    /// Total value references (operand fields, span entries, region
    /// results): the sum of the fan-out array, i.e. the multiplier on
    /// any change to operand width.
    pub value_references: u64,
    pub memory_budget: MemoryBudget,
    pub span_composition: SpanComposition,
    pub timing: PhaseTiming,

    pub value_pool_len: usize,
    pub region_pool_len: usize,
    pub u32_pool_len: usize,
    pub match_arm_pool_len: usize,
    pub interned_types: usize,
    pub interned_signatures: usize,
    pub interned_constants: usize,
    pub globals: usize,

    pub functions: Vec<FunctionCensus>,
}

/// One aggregated CSV row per module, for corpus-level plots.
#[derive(Debug, Serialize)]
pub struct ModuleSummaryRow<'a> {
    pub module: &'a str,
    pub functions: usize,
    pub values: u64,
    pub live_values: u64,
    pub regions: u64,
    pub thetas: u64,
    pub gammas: u64,
    pub candidates: u64,
    pub bail_call: u64,
    pub bail_alias: u64,
    pub bail_nested: u64,
    pub bail_sync: u64,
    pub bail_varying: u64,
    pub foldable: u64,
    pub dup_pure_region: u64,
    pub dup_pure_cross: u64,
    pub licm_movable: u64,
    pub gamma_outputs: u64,
    pub gamma_passthrough: u64,
    pub theta_outputs: u64,
    pub theta_passthrough: u64,
    pub gamma_result_entries: u64,
    pub gamma_poison_results: u64,
    pub dead_projections: u64,
    pub value_references: u64,
    pub fanout_p99: u32,
    pub value_pool: usize,
    pub interned_types: usize,
    pub interned_constants: usize,
    pub bytes_values: usize,
    pub bytes_value_pool: usize,
    pub bytes_regions: usize,
    pub bytes_match_arms: usize,
    pub span_gamma_inputs: usize,
    pub span_region_results: usize,
    pub span_loop_vars: usize,
    pub span_call_args: usize,
    pub span_gep_indices: usize,
    pub span_unaccounted: usize,
    pub frontend_and_parse_ms: u64,
    pub construction_ms: u64,
    pub verify_ms: u64,
    pub census_ms: u64,
}

/// Exhaustive kind naming: a new `ValueKind` variant fails to compile
/// here until the census decides how to count it.
pub fn kind_name(kind: &ValueKind) -> &'static str {
    match kind {
        ValueKind::Const(_) => "Const",
        ValueKind::ConstPoolRef(_) => "ConstPoolRef",
        ValueKind::GlobalRef(_) => "GlobalRef",
        ValueKind::FuncAddr(_) => "FuncAddr",
        ValueKind::Unary { .. } => "Unary",
        ValueKind::Binary { .. } => "Binary",
        ValueKind::ICmp { .. } => "ICmp",
        ValueKind::FCmp { .. } => "FCmp",
        ValueKind::Ternary { .. } => "Ternary",
        ValueKind::Cast { .. } => "Cast",
        ValueKind::ExtractLane { .. } => "ExtractLane",
        ValueKind::InsertLane { .. } => "InsertLane",
        ValueKind::ShuffleLanes { .. } => "ShuffleLanes",
        ValueKind::ExtractField { .. } => "ExtractField",
        ValueKind::InsertField { .. } => "InsertField",
        ValueKind::PtrOffset { .. } => "PtrOffset",
        ValueKind::Load { .. } => "Load",
        ValueKind::Store { .. } => "Store",
        ValueKind::Alloca { .. } => "Alloca",
        ValueKind::AtomicLoad { .. } => "AtomicLoad",
        ValueKind::AtomicStore { .. } => "AtomicStore",
        ValueKind::AtomicReadModifyWrite { .. } => "AtomicReadModifyWrite",
        ValueKind::CompareAndSwap { .. } => "CompareAndSwap",
        ValueKind::Fence { .. } => "Fence",
        ValueKind::Freeze { .. } => "Freeze",
        ValueKind::Match { .. } => "Match",
        ValueKind::Intrinsic { .. } => "Intrinsic",
        ValueKind::Lambda { .. } => "Lambda",
        ValueKind::Theta { .. } => "Theta",
        ValueKind::Gamma { .. } => "Gamma",
        ValueKind::Phi { .. } => "Phi",
        ValueKind::Call { .. } => "Call",
        ValueKind::CallIndirect { .. } => "CallIndirect",
        ValueKind::Project { .. } => "Project",
        ValueKind::RegionParam { .. } => "RegionParam",
        ValueKind::RegionResult { .. } => "RegionResult",
    }
}

/// Pure computations: no state, no structure, safe to deduplicate, fold
/// (operands permitting) or move. Constants and symbol references are
/// handled separately (region-free already).
fn is_pure_compute(kind: &ValueKind) -> bool {
    matches!(
        kind,
        ValueKind::Unary { .. }
            | ValueKind::Binary { .. }
            | ValueKind::ICmp { .. }
            | ValueKind::FCmp { .. }
            | ValueKind::Ternary { .. }
            | ValueKind::Cast { .. }
            | ValueKind::ExtractLane { .. }
            | ValueKind::InsertLane { .. }
            | ValueKind::ShuffleLanes { .. }
            | ValueKind::ExtractField { .. }
            | ValueKind::InsertField { .. }
            | ValueKind::PtrOffset { .. }
            | ValueKind::Freeze { .. }
    )
}

fn is_const_family(kind: &ValueKind) -> bool {
    kind.is_region_free()
}

fn owner_region(owner: &[Owner], value: ValueId) -> Option<u32> {
    match owner[value.0 as usize] {
        Owner::Node { region, .. } | Owner::Param { region } => Some(region),
        Owner::Unowned => None,
    }
}

/// A call's callee is harmless to memory when its declared effects
/// never modify anything (readnone or readonly).
fn effects_are_readonly(attrs: &FnAttrs) -> bool {
    match attrs.memory {
        Some(effects) => {
            let ok = |m: ModRef| matches!(m, ModRef::NoModRef | ModRef::Ref);
            ok(effects.other) && ok(effects.arg_mem) && ok(effects.inaccessible_mem)
        }
        None => false,
    }
}

/// How a region hangs off its parent construct, for tracing region
/// parameters back to the values that feed them.
#[derive(Clone, Copy)]
enum RegionRole {
    ThetaBody {
        theta: ValueId,
    },
    GammaArm {
        gamma: ValueId,
    },
    /// Phi and lambda bodies: parameters are not positionally fed by an
    /// input list this analysis understands.
    Opaque,
}

/// Everything one theta-level analysis needs to know about the regions
/// under a theta body.
struct SubtreeInfo {
    /// Every region under the body (the body itself first), nested
    /// thetas included.
    all: Vec<RegionId>,
    set: FxHashSet<u32>,
    roles: FxHashMap<u32, RegionRole>,
    /// Regions of THIS loop level with their gamma nesting depth: the
    /// body and its transitive gammas, stopping at nested thetas. Depth
    /// 1 is the structural continue/exit demux the construction wraps
    /// every source loop body in, so depth <= 1 is the unconditional
    /// per-iteration shape and depth >= 2 is a source-level conditional.
    own: Vec<(RegionId, u32)>,
    own_depth: FxHashMap<u32, u32>,
    /// Bodies of thetas nested anywhere under this one.
    nested_theta_bodies: Vec<RegionId>,
}

struct Collector<'m> {
    m: &'m RVSDGMod,
    owner: Vec<Owner>,
    // Memoisation for the structural-reduction walks: deeply nested
    // demux chains (interpreter dispatch switches) revisit the same
    // (value, target) and (gamma, slot) queries exponentially often
    // without these. Sound to cache because the graph is acyclic and
    // the answers are context-free. RefCell keeps the walk methods
    // `&self` (they are read-only over the graph; the cells are private
    // caches, and the census is single-threaded).
    traces_memo: std::cell::RefCell<FxHashMap<(ValueId, ValueId), bool>>,
    reduces_memo: std::cell::RefCell<FxHashMap<(u32, ValueId), Option<u32>>>,
    passthrough_memo: std::cell::RefCell<FxHashMap<(RegionId, u32), Option<u32>>>,
}

impl<'m> Collector<'m> {
    /// Walk everything under one theta body. Guarded by the visited
    /// set: a function's Lambda node sits INSIDE its own region, so a
    /// naive walk would re-enqueue the root forever.
    fn subtree_info(&self, theta: ValueId, body: RegionId) -> SubtreeInfo {
        let mut info = SubtreeInfo {
            all: vec![body],
            set: FxHashSet::from_iter([body.0]),
            roles: FxHashMap::from_iter([(body.0, RegionRole::ThetaBody { theta })]),
            own: vec![(body, 0)],
            own_depth: FxHashMap::from_iter([(body.0, 0)]),
            nested_theta_bodies: Vec::new(),
        };
        let mut cursor = 0;
        while cursor < info.all.len() {
            let region_id = info.all[cursor];
            cursor += 1;
            let depth = info.own_depth.get(&region_id.0).copied();
            for &node in &self.m.regions[region_id.0 as usize].nodes {
                match &self.m.values[node.0 as usize].kind {
                    ValueKind::Gamma { regions, .. } => {
                        for &arm in self.m.region_pool.get(*regions) {
                            if info.set.insert(arm.0) {
                                info.all.push(arm);
                                info.roles
                                    .insert(arm.0, RegionRole::GammaArm { gamma: node });
                                if let Some(d) = depth {
                                    info.own.push((arm, d + 1));
                                    info.own_depth.insert(arm.0, d + 1);
                                }
                            }
                        }
                    }
                    ValueKind::Theta {
                        region_id: nested, ..
                    } => {
                        if info.set.insert(nested.0) {
                            info.all.push(*nested);
                            info.roles
                                .insert(nested.0, RegionRole::ThetaBody { theta: node });
                            info.nested_theta_bodies.push(*nested);
                        }
                    }
                    ValueKind::Phi { region, .. } | ValueKind::Lambda { region, .. } => {
                        if info.set.insert(region.0) {
                            info.all.push(*region);
                            info.roles.insert(region.0, RegionRole::Opaque);
                        }
                    }
                    _ => {}
                }
            }
        }
        info
    }

    /// Which parameter POSITION of `region` does `value` structurally
    /// reduce to, seeing through nested gamma outputs that are
    /// themselves pass-through? The position-returning dual of
    /// [`traces_to`](Self::traces_to), used to detect pass-through
    /// output slots.
    fn reduces_to_param(&self, region: RegionId, value: ValueId) -> Option<usize> {
        if let Some(&hit) = self.reduces_memo.borrow().get(&(region.0, value)) {
            return hit.map(|position| position as usize);
        }
        let result = 'walk: {
            let r = &self.m.regions[region.0 as usize];
            if let Some(position) = r.params.iter().position(|&param| param == value) {
                break 'walk Some(position);
            }
            if let ValueKind::Project { call, index } = &self.m.values[value.0 as usize].kind
                && let ValueKind::Gamma {
                    inputs, regions, ..
                } = &self.m.values[call.0 as usize].kind
            {
                let arms = self.m.region_pool.get(*regions);
                let Some(inner_position) = self.gamma_slot_passthrough(arms, *index as usize)
                else {
                    break 'walk None;
                };
                let inputs = self.m.value_pool.get(*inputs);
                let Some(&inner_input) = inputs.get(inner_position) else {
                    break 'walk None;
                };
                break 'walk self.reduces_to_param(region, inner_input);
            }
            None
        };
        self.reduces_memo
            .borrow_mut()
            .insert((region.0, value), result.map(|position| position as u32));
        result
    }

    /// Is output slot `slot` of a gamma pure freight: does EVERY arm's
    /// result in that slot reduce to the arm's own parameter at one
    /// agreed position? If so the output equals the gamma input at that
    /// position, and the slot exists only to carry the value through.
    fn gamma_slot_passthrough(&self, arms: &[RegionId], slot: usize) -> Option<usize> {
        let memo_key = (*arms.first()?, slot as u32);
        if let Some(&hit) = self.passthrough_memo.borrow().get(&memo_key) {
            return hit.map(|position| position as usize);
        }
        let mut agreed: Option<usize> = None;
        for &arm in arms {
            let region = &self.m.regions[arm.0 as usize];
            let results = self.m.value_pool.get(region.results);
            let position = results
                .get(slot)
                .and_then(|&result| self.reduces_to_param(arm, result));
            match (position, agreed) {
                (None, _) => {
                    agreed = None;
                    break;
                }
                (Some(found), None) => agreed = Some(found),
                (Some(found), Some(previous)) if previous == found => {}
                (Some(_), Some(_)) => {
                    agreed = None;
                    break;
                }
            }
        }
        self.passthrough_memo
            .borrow_mut()
            .insert(memo_key, agreed.map(|position| position as u32));
        agreed
    }

    /// Does `value` structurally reduce to `target` (a region param of
    /// the same region)? Sees through gamma outputs where EVERY arm
    /// yields its own image of the target (the arm parameter fed by the
    /// target). This is how a loop variable that merely passes through
    /// the construction's continue/exit demux is recognised as
    /// unchanged: Reissmann's invariant value redirection.
    fn traces_to(&self, value: ValueId, target: ValueId) -> bool {
        if value == target {
            return true;
        }
        if let Some(&hit) = self.traces_memo.borrow().get(&(value, target)) {
            return hit;
        }
        let result = 'walk: {
            let ValueKind::Project { call, index } = &self.m.values[value.0 as usize].kind else {
                break 'walk false;
            };
            let ValueKind::Gamma {
                inputs, regions, ..
            } = &self.m.values[call.0 as usize].kind
            else {
                break 'walk false;
            };
            let inputs = self.m.value_pool.get(*inputs);
            let Some(position) = inputs
                .iter()
                .position(|&input| self.traces_to(input, target))
            else {
                break 'walk false;
            };
            self.m.region_pool.get(*regions).iter().all(|&arm_region| {
                let arm = &self.m.regions[arm_region.0 as usize];
                let (Some(&arm_target), Some(&arm_result)) = (
                    arm.params.get(position),
                    self.m.value_pool.get(arm.results).get(*index as usize),
                ) else {
                    return false;
                };
                self.traces_to(arm_result, arm_target)
            })
        };
        self.traces_memo
            .borrow_mut()
            .insert((value, target), result);
        result
    }

    /// Is `value` invariant with respect to a theta body subtree: a
    /// constant/symbol, defined outside the subtree, a pure computation
    /// of invariants, or a region parameter fed only by invariants (a
    /// gamma capture of one, or a theta loop variable redirected
    /// through the body unchanged). This is the loop-invariance rule
    /// the motion pass will reuse.
    fn invariant_in(
        &self,
        info: &SubtreeInfo,
        memo: &mut FxHashMap<ValueId, bool>,
        value: ValueId,
    ) -> bool {
        if let Some(&hit) = memo.get(&value) {
            return hit;
        }
        // Break recursion through redirection cycles (a loop var whose
        // invariance is being decided must not consult itself).
        memo.insert(value, false);
        let kind = &self.m.values[value.0 as usize].kind;
        let result = if is_const_family(kind) {
            true
        } else {
            match owner_region(&self.owner, value) {
                Some(region) if !info.set.contains(&region) => true,
                None => false,
                Some(region) => {
                    if is_pure_compute(kind) {
                        let mut operands = Vec::new();
                        self.m.for_each_value_operand(value, |op| operands.push(op));
                        operands
                            .into_iter()
                            .all(|op| self.invariant_in(info, memo, op))
                    } else if let ValueKind::RegionParam { index, .. } = kind {
                        let index = *index as usize;
                        match info.roles.get(&region) {
                            Some(RegionRole::GammaArm { gamma }) => {
                                let ValueKind::Gamma { inputs, .. } =
                                    &self.m.values[gamma.0 as usize].kind
                                else {
                                    return false;
                                };
                                let inputs = self.m.value_pool.get(*inputs);
                                index < inputs.len() && self.invariant_in(info, memo, inputs[index])
                            }
                            Some(RegionRole::ThetaBody { theta }) => {
                                let ValueKind::Theta { loop_vars, .. } =
                                    &self.m.values[theta.0 as usize].kind
                                else {
                                    return false;
                                };
                                let body = &self.m.regions[region as usize];
                                let results = self.m.value_pool.get(body.results);
                                let loop_vars = self.m.value_pool.get(*loop_vars);
                                index < results.len()
                                    && index < loop_vars.len()
                                    && self.traces_to(results[index], value)
                                    && self.invariant_in(info, memo, loop_vars[index])
                            }
                            Some(RegionRole::Opaque) | None => false,
                        }
                    } else {
                        false
                    }
                }
            }
        };
        memo.insert(value, result);
        result
    }

    /// Memory/state census, the loop-invariant-motion dry run, and the
    /// scalar-promotion dry run for one theta.
    fn theta_body_census(&self, theta: ValueId, body: RegionId, fc: &mut FunctionCensus) {
        let info = self.subtree_info(theta, body);
        let mut memo: FxHashMap<ValueId, bool> = FxHashMap::default();

        // Traffic census over the WHOLE subtree (nested loops included),
        // resolving every access once for reuse below.
        let mut mem_ops: u32 = 0;
        let mut bases: FxHashSet<BaseObject> = FxHashSet::default();
        let mut resolved: FxHashMap<ValueId, crate::rvsdg::alias::ResolvedAddress> =
            FxHashMap::default();
        for &region_id in &info.all {
            for &node in &self.m.regions[region_id.0 as usize].nodes {
                match &self.m.values[node.0 as usize].kind {
                    ValueKind::Load { addr, .. }
                    | ValueKind::Store { addr, .. }
                    | ValueKind::AtomicLoad { addr, .. }
                    | ValueKind::AtomicStore { addr, .. }
                    | ValueKind::AtomicReadModifyWrite { addr, .. }
                    | ValueKind::CompareAndSwap { addr, .. } => {
                        mem_ops += 1;
                        let address = resolved
                            .entry(*addr)
                            .or_insert_with(|| self.m.resolve_address(*addr));
                        bases.insert(address.base);
                    }
                    ValueKind::Fence { .. } => mem_ops += 1,
                    ValueKind::Call { fn_id, .. } => {
                        fc.calls_in_thetas += 1;
                        if effects_are_readonly(&self.m.get_function(*fn_id).attrs) {
                            fc.calls_readonly += 1;
                        }
                    }
                    ValueKind::CallIndirect { .. } => fc.calls_in_thetas += 1,
                    _ => {}
                }
            }
        }
        fc.theta_mem_ops.push(mem_ops);
        fc.theta_distinct_bases.push(bases.len() as u32);

        // This loop level's regions: address-origin classification and
        // the invariant-motion dry run.
        for &(region_id, _depth) in &info.own {
            for &node in &self.m.regions[region_id.0 as usize].nodes {
                let kind = &self.m.values[node.0 as usize].kind;
                match kind {
                    ValueKind::Load { addr, .. } | ValueKind::Store { addr, .. } => {
                        let external = is_const_family(&self.m.values[addr.0 as usize].kind)
                            || owner_region(&self.owner, *addr)
                                .is_some_and(|region| !info.set.contains(&region));
                        if external {
                            fc.addr_external += 1;
                        } else if self.invariant_in(&info, &mut memo, *addr) {
                            fc.addr_computable += 1;
                        } else {
                            fc.addr_varying += 1;
                        }
                    }
                    _ => {}
                }
                if is_pure_compute(kind) {
                    let mut operands = Vec::new();
                    self.m.for_each_value_operand(node, |op| operands.push(op));
                    if !operands.is_empty()
                        && operands
                            .iter()
                            .all(|&op| self.invariant_in(&info, &mut memo, op))
                    {
                        fc.licm_movable += 1;
                    }
                }
            }
        }

        // Promotion dry run: one verdict per store at this loop level.
        // Depth <= 1 (the body or the structural continue/exit demux
        // arm) is the unconditional per-iteration shape; depth >= 2 is
        // a source-level conditional and counts as nested.
        for &(store_region, store_depth) in &info.own {
            for &store in &self.m.regions[store_region.0 as usize].nodes {
                let (cell_addr, cell_volatile) = match &self.m.values[store.0 as usize].kind {
                    ValueKind::Store { addr, volatile, .. } => (*addr, *volatile),
                    _ => continue,
                };
                if !self.invariant_in(&info, &mut memo, cell_addr) {
                    fc.bail_varying += 1;
                    continue;
                }
                let cell = &resolved[&cell_addr];
                let mut call = false;
                let mut alias = false;
                let mut nested = store_depth >= 2;
                let mut sync = cell_volatile;
                for &region_id in &info.all {
                    let depth = info.own_depth.get(&region_id.0).copied();
                    for &node in &self.m.regions[region_id.0 as usize].nodes {
                        if node == store {
                            continue;
                        }
                        match &self.m.values[node.0 as usize].kind {
                            ValueKind::Call { fn_id, .. } => {
                                if !effects_are_readonly(&self.m.get_function(*fn_id).attrs) {
                                    call = true;
                                }
                            }
                            ValueKind::CallIndirect { .. } => call = true,
                            ValueKind::Intrinsic { op, .. } => {
                                if matches!(
                                    op,
                                    IntrinsicOp::MemCopy
                                        | IntrinsicOp::MemMove
                                        | IntrinsicOp::MemSet
                                ) {
                                    call = true;
                                }
                            }
                            ValueKind::AtomicLoad { .. }
                            | ValueKind::AtomicStore { .. }
                            | ValueKind::AtomicReadModifyWrite { .. }
                            | ValueKind::CompareAndSwap { .. }
                            | ValueKind::Fence { .. } => sync = true,
                            ValueKind::Load { addr, volatile, .. }
                            | ValueKind::Store { addr, volatile, .. } => {
                                let address = &resolved[addr];
                                // The same address VALUE is the same
                                // cell even when its offsets are
                                // runtime values (the invariance check
                                // already passed for it).
                                if *addr == cell_addr || cell.same_cell(address) {
                                    // Same cell: fine at the
                                    // unconditional depths of this loop
                                    // level, nested anywhere else (a
                                    // source conditional or an inner
                                    // loop).
                                    if !matches!(depth, Some(d) if d <= 1) {
                                        nested = true;
                                    }
                                    if *volatile {
                                        sync = true;
                                    }
                                } else if may_alias_resolved(cell, address) {
                                    alias = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if call || alias || nested || sync {
                    fc.bail_call += call as u64;
                    fc.bail_alias += alias as u64;
                    fc.bail_nested += nested as u64;
                    fc.bail_sync += sync as u64;
                } else {
                    fc.promotion_candidates += 1;
                }
            }
        }
    }

    fn function_census(&self, name: String, root: RegionId) -> FunctionCensus {
        let mut fc = FunctionCensus {
            name,
            ..Default::default()
        };
        let mut theta_bodies: Vec<(ValueId, RegionId)> = Vec::new();
        let mut cross_region_seen: FxHashMap<&Value, u32> = FxHashMap::default();

        // Visited guard: a function's Lambda node sits INSIDE its own
        // region, so the walk would otherwise revisit the root forever.
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        seen.insert(root.0);
        let mut stack: Vec<(RegionId, u32, u32)> = vec![(root, 0, 0)];
        while let Some((region_id, depth, loop_depth)) = stack.pop() {
            let region = &self.m.regions[region_id.0 as usize];
            fc.regions += 1;
            fc.region_node_counts.push(region.nodes.len() as u32);
            fc.values += (region.nodes.len() + region.params.len()) as u64;
            fc.max_depth = fc.max_depth.max(depth);
            fc.max_loop_depth = fc.max_loop_depth.max(loop_depth);

            let mut in_region_seen: FxHashSet<&Value> = FxHashSet::default();
            for &node in &region.nodes {
                let value = &self.m.values[node.0 as usize];
                match &value.kind {
                    ValueKind::Gamma { regions, .. } => {
                        let arms = self.m.region_pool.get(*regions);
                        fc.gammas += 1;
                        fc.gamma_arities.push(arms.len() as u32);
                        let result_count = arms
                            .first()
                            .map(|&arm| {
                                self.m
                                    .value_pool
                                    .get(self.m.regions[arm.0 as usize].results)
                                    .len()
                            })
                            .unwrap_or(0);
                        fc.gamma_outputs += result_count as u64;
                        for slot in 0..result_count {
                            if self.gamma_slot_passthrough(arms, slot).is_some() {
                                fc.gamma_passthrough += 1;
                            }
                        }
                        for &arm in arms {
                            let arm_region = &self.m.regions[arm.0 as usize];
                            for &result in self.m.value_pool.get(arm_region.results) {
                                fc.gamma_result_entries += 1;
                                if matches!(
                                    self.m.values[result.0 as usize].kind,
                                    ValueKind::Const(ConstValue::Poison)
                                ) {
                                    fc.gamma_poison_results += 1;
                                }
                            }
                            if seen.insert(arm.0) {
                                stack.push((arm, depth + 1, loop_depth));
                            }
                        }
                    }
                    ValueKind::Theta {
                        region_id: body,
                        loop_vars,
                        ..
                    } => {
                        fc.thetas += 1;
                        fc.theta_arities
                            .push(self.m.value_pool.get(*loop_vars).len() as u32);
                        let body_region = &self.m.regions[body.0 as usize];
                        let results = self.m.value_pool.get(body_region.results);
                        fc.theta_outputs += results.len() as u64;
                        for (slot, &result) in results.iter().enumerate() {
                            if let Some(&param) = body_region.params.get(slot)
                                && self.traces_to(result, param)
                            {
                                fc.theta_passthrough += 1;
                            }
                        }
                        theta_bodies.push((node, *body));
                        if seen.insert(body.0) {
                            stack.push((*body, depth + 1, loop_depth + 1));
                        }
                    }
                    ValueKind::Phi { region, .. } | ValueKind::Lambda { region, .. } => {
                        if seen.insert(region.0) {
                            stack.push((*region, depth + 1, loop_depth));
                        }
                    }
                    ValueKind::Match { .. } => fc.matches += 1,
                    ValueKind::Project { .. } => fc.projections += 1,
                    _ => {}
                }

                // CNE dry run: identical pure/constant values. Spans are
                // compared by id, so structurally equal nodes with
                // separately pooled spans undercount; conservative.
                if is_pure_compute(&value.kind) || is_const_family(&value.kind) {
                    if !in_region_seen.insert(value) {
                        fc.dup_pure_in_region += 1;
                    }
                    let seen = cross_region_seen.entry(value).or_insert(0);
                    if *seen > 0 {
                        fc.dup_pure_cross_region += 1;
                    }
                    *seen += 1;
                }

                // Fold dry run: pure compute over constants only.
                if matches!(
                    value.kind,
                    ValueKind::Unary { .. }
                        | ValueKind::Binary { .. }
                        | ValueKind::ICmp { .. }
                        | ValueKind::FCmp { .. }
                        | ValueKind::Cast { .. }
                        | ValueKind::Ternary { .. }
                ) {
                    let mut all_const = true;
                    self.m.for_each_value_operand(node, |op| {
                        all_const &= matches!(
                            self.m.values[op.0 as usize].kind,
                            ValueKind::Const(_) | ValueKind::ConstPoolRef(_)
                        );
                    });
                    if all_const {
                        fc.foldable += 1;
                    }
                }
            }
        }
        // Cross-region count includes the in-region duplicates; report
        // it as duplicates in ADDITION to those.
        fc.dup_pure_cross_region = fc
            .dup_pure_cross_region
            .saturating_sub(fc.dup_pure_in_region);

        // Memory census + invariant-motion and promotion dry runs, one
        // theta at a time.
        for (theta, body) in theta_bodies {
            self.theta_body_census(theta, body, &mut fc);
        }
        fc
    }
}

pub fn collect(m: &RVSDGMod) -> ModuleCensus {
    let mut ownership_errs = Vec::new();
    let collector = Collector {
        owner: m.build_value_ownership(&mut ownership_errs),
        m,
        traces_memo: Default::default(),
        reduces_memo: Default::default(),
        passthrough_memo: Default::default(),
    };

    let mut census = ModuleCensus {
        mod_name: m.mod_name.clone(),
        total_values: m.values.len() as u64,
        value_pool_len: m.value_pool.len(),
        region_pool_len: m.region_pool.len(),
        u32_pool_len: m.u32_pool.len(),
        match_arm_pool_len: m.match_arm_pool.len(),
        interned_types: m.types.interned_len(),
        interned_signatures: m.signatures.len(),
        interned_constants: m.constants.len(),
        globals: m.globals.len(),
        ..Default::default()
    };

    // Kind histogram + projection parents.
    for value in &m.values {
        *census
            .kind_counts
            .entry(kind_name(&value.kind))
            .or_insert(0) += 1;
        if let ValueKind::Project { call, .. } = &value.kind {
            let parent = kind_name(&m.values[call.0 as usize].kind);
            *census.projections_by_parent.entry(parent).or_insert(0) += 1;
        }
    }

    // Fan-out: value-operand uses plus region results (skipping
    // RegionResult values, whose span IS the owning region's results).
    census.fanout = vec![0u32; m.values.len()];
    for (index, value) in m.values.iter().enumerate() {
        if matches!(value.kind, ValueKind::RegionResult { .. }) {
            continue;
        }
        m.for_each_value_operand(ValueId(index as u32), |op| {
            census.fanout[op.0 as usize] += 1;
        });
    }
    for region in &m.regions {
        for &result in m.value_pool.get(region.results) {
            census.fanout[result.0 as usize] += 1;
        }
    }
    census.dead_projections = m
        .values
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            matches!(value.kind, ValueKind::Project { .. }) && census.fanout[*index] == 0
        })
        .count() as u64;
    census.value_references = census.fanout.iter().map(|&uses| uses as u64).sum();

    // Byte budget of the backing arrays.
    census.memory_budget = MemoryBudget {
        values_bytes: m.values.len() * size_of::<Value>(),
        value_pool_bytes: m.value_pool.len() * size_of::<ValueId>(),
        region_structs_bytes: m.regions.len() * size_of::<crate::rvsdg::Region>(),
        region_lists_bytes: m
            .regions
            .iter()
            .map(|region| (region.params.len() + region.nodes.len()) * size_of::<ValueId>())
            .sum(),
        region_pool_bytes: m.region_pool.len() * size_of::<RegionId>(),
        u32_pool_bytes: m.u32_pool.len() * size_of::<u32>(),
        match_arm_pool_bytes: m.match_arm_pool.len() * size_of::<crate::rvsdg::MatchArm>(),
    };

    // value_pool composition: which span kinds fill it.
    let mut spans = SpanComposition::default();
    for value in &m.values {
        match &value.kind {
            ValueKind::Gamma { inputs, .. } => spans.gamma_inputs += inputs.len as usize,
            ValueKind::Theta { loop_vars, .. } => spans.theta_loop_vars += loop_vars.len as usize,
            ValueKind::Call { args, .. }
            | ValueKind::CallIndirect { args, .. }
            | ValueKind::Intrinsic { args, .. } => spans.call_args += args.len as usize,
            ValueKind::PtrOffset { indices, .. } => {
                spans.ptr_offset_indices += indices.len as usize
            }
            ValueKind::ShuffleLanes { mask, .. } => spans.shuffle_masks += mask.len as usize,
            _ => {}
        }
    }
    for region in &m.regions {
        spans.region_results += region.results.len as usize;
    }
    spans.unaccounted = m.value_pool.len().saturating_sub(
        spans.gamma_inputs
            + spans.region_results
            + spans.theta_loop_vars
            + spans.call_args
            + spans.ptr_offset_indices
            + spans.shuffle_masks,
    );
    census.span_composition = spans;

    let mut ranked: Vec<usize> = (0..m.values.len()).collect();
    ranked.sort_unstable_by_key(|&index| std::cmp::Reverse(census.fanout[index]));
    census.top_fanout = ranked
        .into_iter()
        .take(10)
        .filter(|&index| census.fanout[index] > 0)
        .map(|index| {
            (
                ValueId(index as u32),
                kind_name(&m.values[index].kind),
                census.fanout[index],
            )
        })
        .collect();

    // Liveness: region membership seeds a value-operand closure. The gap
    // to total_values is the husk fraction passes leave behind.
    let mut live = vec![false; m.values.len()];
    let mut worklist: Vec<ValueId> = Vec::new();
    let mark = |live: &mut Vec<bool>, worklist: &mut Vec<ValueId>, value: ValueId| {
        if !live[value.0 as usize] {
            live[value.0 as usize] = true;
            worklist.push(value);
        }
    };
    for region in &m.regions {
        mark(&mut live, &mut worklist, region.entry_state.0);
        for &param in &region.params {
            mark(&mut live, &mut worklist, param);
        }
        for &node in &region.nodes {
            mark(&mut live, &mut worklist, node);
        }
        for &result in m.value_pool.get(region.results) {
            mark(&mut live, &mut worklist, result);
        }
    }
    for (index, value) in m.values.iter().enumerate() {
        if matches!(value.kind, ValueKind::RegionResult { .. }) {
            mark(&mut live, &mut worklist, ValueId(index as u32));
        }
    }
    let mut operands = Vec::new();
    while let Some(value) = worklist.pop() {
        operands.clear();
        m.for_each_value_operand(value, |op| operands.push(op));
        for &op in &operands {
            mark(&mut live, &mut worklist, op);
        }
    }
    census.live_values = live.iter().filter(|&&l| l).count() as u64;

    // Per-function region trees.
    for function in &m.functions {
        let Some(lambda) = function.lambda_val else {
            continue;
        };
        let ValueKind::Lambda { region, .. } = &m.values[lambda.0 as usize].kind else {
            continue;
        };
        census
            .functions
            .push(collector.function_census(function.name.clone(), *region));
    }

    census
}

/// p-th percentile of an unsorted sample (p in 0..=100).
pub fn percentile(samples: &[u32], p: u32) -> u32 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (p as usize * (sorted.len() - 1)).div_ceil(100);
    sorted[rank.min(sorted.len() - 1)]
}

impl ModuleCensus {
    pub fn summary_row(&self) -> ModuleSummaryRow<'_> {
        let sum = |f: fn(&FunctionCensus) -> u64| self.functions.iter().map(f).sum::<u64>();
        ModuleSummaryRow {
            module: &self.mod_name,
            functions: self.functions.len(),
            values: self.total_values,
            live_values: self.live_values,
            regions: sum(|f| f.regions),
            thetas: sum(|f| f.thetas),
            gammas: sum(|f| f.gammas),
            candidates: sum(|f| f.promotion_candidates),
            bail_call: sum(|f| f.bail_call),
            bail_alias: sum(|f| f.bail_alias),
            bail_nested: sum(|f| f.bail_nested),
            bail_sync: sum(|f| f.bail_sync),
            bail_varying: sum(|f| f.bail_varying),
            foldable: sum(|f| f.foldable),
            dup_pure_region: sum(|f| f.dup_pure_in_region),
            dup_pure_cross: sum(|f| f.dup_pure_cross_region),
            licm_movable: sum(|f| f.licm_movable),
            gamma_outputs: sum(|f| f.gamma_outputs),
            gamma_passthrough: sum(|f| f.gamma_passthrough),
            theta_outputs: sum(|f| f.theta_outputs),
            theta_passthrough: sum(|f| f.theta_passthrough),
            gamma_result_entries: sum(|f| f.gamma_result_entries),
            gamma_poison_results: sum(|f| f.gamma_poison_results),
            dead_projections: self.dead_projections,
            value_references: self.value_references,
            fanout_p99: percentile(&self.fanout, 99),
            value_pool: self.value_pool_len,
            interned_types: self.interned_types,
            interned_constants: self.interned_constants,
            bytes_values: self.memory_budget.values_bytes,
            bytes_value_pool: self.memory_budget.value_pool_bytes,
            bytes_regions: self.memory_budget.region_structs_bytes
                + self.memory_budget.region_lists_bytes
                + self.memory_budget.region_pool_bytes,
            bytes_match_arms: self.memory_budget.match_arm_pool_bytes,
            span_gamma_inputs: self.span_composition.gamma_inputs,
            span_region_results: self.span_composition.region_results,
            span_loop_vars: self.span_composition.theta_loop_vars,
            span_call_args: self.span_composition.call_args,
            span_gep_indices: self.span_composition.ptr_offset_indices,
            span_unaccounted: self.span_composition.unaccounted,
            frontend_and_parse_ms: self.timing.frontend_and_parse_ms,
            construction_ms: self.timing.construction_ms,
            verify_ms: self.timing.verify_ms,
            census_ms: self.timing.census_ms,
        }
    }

    /// Human summary. The caller supplies the sink (a buffered stdout
    /// lock from the driver; a Vec in tests).
    pub fn write_summary(&self, out: &mut impl io::Write) -> io::Result<()> {
        let summary = self.summary_row();
        writeln!(out, "=== module {} ===", self.mod_name)?;
        writeln!(
            out,
            "values {} (live {}, dead {}), functions {}, globals {}",
            self.total_values,
            self.live_values,
            self.total_values - self.live_values,
            self.functions.len(),
            self.globals,
        )?;
        writeln!(
            out,
            "pools: values {}, regions {}, u32 {}, match arms {}; interned: types {}, signatures {}, constants {}",
            self.value_pool_len,
            self.region_pool_len,
            self.u32_pool_len,
            self.match_arm_pool_len,
            self.interned_types,
            self.interned_signatures,
            self.interned_constants,
        )?;

        let mut kinds: Vec<(&&str, &u64)> = self.kind_counts.iter().collect();
        kinds.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(**count));
        writeln!(out, "-- value kinds --")?;
        for (name, count) in kinds {
            writeln!(out, "  {name:<22} {count}")?;
        }
        if !self.projections_by_parent.is_empty() {
            let mut parents: Vec<(&&str, &u64)> = self.projections_by_parent.iter().collect();
            parents.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(**count));
            writeln!(out, "-- projections by parent --")?;
            for (name, count) in parents {
                writeln!(out, "  {name:<22} {count}")?;
            }
        }

        writeln!(
            out,
            "-- fan-out -- p50 {} p99 {} max {}",
            percentile(&self.fanout, 50),
            percentile(&self.fanout, 99),
            self.fanout.iter().copied().max().unwrap_or(0),
        )?;
        for (value, kind, uses) in &self.top_fanout {
            writeln!(out, "  {value:?} {kind:<14} {uses} uses")?;
        }

        writeln!(out, "-- structure --")?;
        writeln!(
            out,
            "  regions {}  thetas {}  gammas {}  max depth {}  max loop depth {}",
            summary.regions,
            summary.thetas,
            summary.gammas,
            self.functions
                .iter()
                .map(|f| f.max_depth)
                .max()
                .unwrap_or(0),
            self.functions
                .iter()
                .map(|f| f.max_loop_depth)
                .max()
                .unwrap_or(0),
        )?;
        writeln!(out, "-- opportunities --")?;
        writeln!(
            out,
            "  foldable {}  dup pure (in-region {} / cross {})  licm movable {}",
            summary.foldable, summary.dup_pure_region, summary.dup_pure_cross, summary.licm_movable,
        )?;
        writeln!(
            out,
            "  promotion candidates {}  bails: call {} alias {} nested {} sync {} varying {}",
            summary.candidates,
            summary.bail_call,
            summary.bail_alias,
            summary.bail_nested,
            summary.bail_sync,
            summary.bail_varying,
        )?;
        let sum = |f: fn(&FunctionCensus) -> u64| self.functions.iter().map(f).sum::<u64>();
        writeln!(
            out,
            "  pass-through output slots: gamma {}/{} theta {}/{}; dead projections {}",
            summary.gamma_passthrough,
            summary.gamma_outputs,
            summary.theta_passthrough,
            summary.theta_outputs,
            self.dead_projections,
        )?;
        writeln!(
            out,
            "  gamma arm results: {} entries, {} poison",
            summary.gamma_result_entries, summary.gamma_poison_results,
        )?;
        writeln!(
            out,
            "  value references (fan-out sum): {}",
            self.value_references
        )?;
        let budget = &self.memory_budget;
        writeln!(
            out,
            "-- memory budget -- values {}KB, value_pool {}KB, regions {}KB (structs {} + lists {} + pool {}), match arms {}KB",
            budget.values_bytes / 1024,
            budget.value_pool_bytes / 1024,
            (budget.region_structs_bytes + budget.region_lists_bytes + budget.region_pool_bytes)
                / 1024,
            budget.region_structs_bytes / 1024,
            budget.region_lists_bytes / 1024,
            budget.region_pool_bytes / 1024,
            budget.match_arm_pool_bytes / 1024,
        )?;
        let spans = &self.span_composition;
        writeln!(
            out,
            "-- value_pool spans -- gamma inputs {}, region results {}, loop vars {}, call args {}, gep indices {}, shuffle masks {}, unaccounted {}",
            spans.gamma_inputs,
            spans.region_results,
            spans.theta_loop_vars,
            spans.call_args,
            spans.ptr_offset_indices,
            spans.shuffle_masks,
            spans.unaccounted,
        )?;
        if self.timing.frontend_and_parse_ms
            + self.timing.construction_ms
            + self.timing.verify_ms
            + self.timing.census_ms
            > 0
        {
            writeln!(
                out,
                "-- timing -- frontend+parse {}ms, construction {}ms, verify {}ms, census {}ms",
                self.timing.frontend_and_parse_ms,
                self.timing.construction_ms,
                self.timing.verify_ms,
                self.timing.census_ms,
            )?;
        }
        writeln!(
            out,
            "  addr origin in theta bodies: external {} computable {} varying {}",
            sum(|f| f.addr_external),
            sum(|f| f.addr_computable),
            sum(|f| f.addr_varying),
        )?;
        writeln!(
            out,
            "  calls in theta bodies {} (readonly effects {})",
            sum(|f| f.calls_in_thetas),
            sum(|f| f.calls_readonly),
        )?;

        let mut largest: Vec<&FunctionCensus> = self.functions.iter().collect();
        largest.sort_unstable_by_key(|f| std::cmp::Reverse(f.values));
        writeln!(out, "-- largest functions --")?;
        for f in largest.iter().take(5) {
            writeln!(
                out,
                "  {:<32} values {} regions {} thetas {} candidates {}",
                f.name, f.values, f.regions, f.thetas, f.promotion_candidates
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ICmpPred, Linkage, RVSDGMod,
        builder::LoopResult,
        func::FnResult,
        types::{I32, PtrType, TypeRef},
    };

    fn ptr_ty(rvsdg: &mut RVSDGMod) -> TypeRef {
        let id = rvsdg.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        });
        TypeRef::Ptr(id)
    }

    /// The promotable shape: a theta body accumulating through a cell
    /// whose address (an alloca in the parent region) is invariant.
    #[test]
    fn census_counts_promotion_candidate() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut rvsdg);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb, state| {
                let one = rb.const_i32(1);
                let alloc = rb.alloca(state, I32, one, ptr);
                let zero = rb.const_i32(0);
                let s1 = rb.store(alloc.state, alloc.ptr, zero, None, false);
                let i = rb.const_i32(0);
                let res = rb.theta(s1, &[i], |rb| {
                    let loop_i = rb.param(0);
                    let cur = rb.load(s1, alloc.ptr, I32, None, false);
                    let ten = rb.const_i32(10);
                    let next_val =
                        rb.binary(BinaryOp::Add, ArithFlags::default(), cur.value, ten, I32);
                    let s2 = rb.store(cur.state, alloc.ptr, next_val, None, false);
                    let one = rb.const_i32(1);
                    let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                    let five = rb.const_i32(5);
                    let cond = rb.icmp(ICmpPred::SignedLt, next_i, five);
                    Ok(LoopResult {
                        condition: cond,
                        next_state: s2,
                        next_vars: vec![next_i],
                    })
                })?;
                let final_val = rb.load(res.state, alloc.ptr, I32, None, false);
                Ok(FnResult {
                    state: final_val.state,
                    values: vec![final_val.value],
                })
            })
            .unwrap();

        let census = collect(&rvsdg);
        assert_eq!(census.functions.len(), 1);
        let f = &census.functions[0];
        assert_eq!(f.thetas, 1);
        assert_eq!(f.gammas, 0);
        assert_eq!(f.promotion_candidates, 1, "census: {f:?}");
        assert_eq!(
            f.bail_call + f.bail_alias + f.bail_nested + f.bail_sync + f.bail_varying,
            0,
            "census: {f:?}"
        );
        assert_eq!(f.max_loop_depth, 1);
        // Everything constructed is reachable from a region.
        assert_eq!(census.total_values, census.live_values);
    }

    /// A store whose address is created INSIDE the body (an alloca in
    /// the loop) has nothing invariant to carry: bail_varying.
    #[test]
    fn census_counts_varying_address_bail() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut rvsdg);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb, state| {
                let i = rb.const_i32(0);
                let res = rb.theta(state, &[i], |rb| {
                    let loop_i = rb.param(0);
                    let one = rb.const_i32(1);
                    let alloc = rb.alloca(state, I32, one, ptr);
                    let s2 = rb.store(alloc.state, alloc.ptr, loop_i, None, false);
                    let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                    let five = rb.const_i32(5);
                    let cond = rb.icmp(ICmpPred::SignedLt, next_i, five);
                    Ok(LoopResult {
                        condition: cond,
                        next_state: s2,
                        next_vars: vec![next_i],
                    })
                })?;
                let zero = rb.const_i32(0);
                Ok(FnResult {
                    state: res.state,
                    values: vec![zero],
                })
            })
            .unwrap();

        let census = collect(&rvsdg);
        let f = &census.functions[0];
        assert_eq!(f.promotion_candidates, 0, "census: {f:?}");
        assert_eq!(f.bail_varying, 1, "census: {f:?}");
    }
}
