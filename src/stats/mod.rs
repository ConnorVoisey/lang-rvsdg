//! Read-only census over a constructed RVSDG.
//!
//! Two jobs: design input for the optimizer substrate (rebuild vs
//! mutate, holes vs compaction, use-list strategy, how far alias
//! analysis must go) and a before/after shape diff that quantifies each
//! pass and catches hygiene regressions correctness tests cannot see.
//! Surfaced through the compiler's --stats / --stats-json flags.
//!
//! Several queries here are deliberate DRY RUNS of pass analyses -- the
//! duplicate-node map (common node elimination), the foldable count
//! (constant folding), the invariance test (loop-invariant motion), the
//! promotion candidate scan -- so the census and the eventual passes
//! share one implementation of each rule. Base-object resolution lives
//! in `rvsdg::alias` for the same reason.

pub mod heap;

use std::io;
use std::mem::size_of;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;

use crate::rvsdg::{
    ConstValue, RVSDGMod, RegionId, Value, ValueId, ValueKind,
    alias::{BaseObject, may_alias_resolved},
    func::{FnAttrs, ModRef},
    function_graph::FunctionGraph,
    ops::IntrinsicOp,
    types::TypeRef,
    verify::scope::Owner,
};

/// Census of one function's region tree. Distributions keep their raw
/// samples; the human summary derives percentiles and maxima from them.
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
    /// The regions' interface and node blocks, which live INSIDE the
    /// value pool: a SUBSET of `value_pool_bytes` broken out for the
    /// region view, never added to other byte figures.
    pub region_blocks_bytes: usize,
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
    /// Every region's params segment (interface blocks live in the pool).
    pub region_params: usize,
    /// Every region's results segment (function bodies, gamma arms,
    /// theta bodies alike).
    pub region_results: usize,
    /// Every region's nodes block.
    pub region_nodes: usize,
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
    pub frontend_and_parse_ms: f64,
    pub construction_ms: f64,
    /// Everything spent verifying: the debug post-construction check
    /// plus the --verify-all pipeline checks. Absent verification
    /// reports as zero.
    pub verify_ms: f64,
    pub optimise_ms: f64,
    /// Total across every census snapshot taken this compile.
    pub census_ms: f64,
}

/// Whole-compile statistics document written by `--stats-json`: the
/// census summaries before and after the pass pipeline, one row per
/// pass, and the emitted-IR counts. One JSON object per compile, for
/// corpus sweeps and regression tracking.
#[derive(Debug, Serialize)]
pub struct CompileReportJson<'a> {
    /// Bumped on any breaking change to this document's shape, so
    /// downstream tooling rejects or adapts instead of misparsing.
    pub schema_version: u32,
    pub input: &'a str,
    /// Per-compile facts, stated once (census rows carry graph shape
    /// only).
    pub phases: PhaseTiming,
    pub heap: HeapUsage,
    pub census_pre_opt: &'a ModuleSummaryRow,
    pub census_post_opt: Option<&'a ModuleSummaryRow>,
    pub passes: &'a [crate::opt::PassReport],
    pub emitted_ir: Option<EmittedIrStats>,
    /// Lowering + codegen + link (or JIT engine build), everything
    /// after the pass pipeline; None when that stage was never reached.
    pub output_ms: Option<f64>,
    /// The compile failure this document accompanies, if any: stats are
    /// written even for failed compiles (write-what-you-have), and this
    /// field is how a partial document is told apart from a complete one.
    pub error: Option<&'a str>,
}

/// The current [`CompileReportJson`] shape version.
pub const COMPILE_REPORT_SCHEMA_VERSION: u32 = 1;

/// Shape of the emitted LLVM module, counted by a walk after lowering.
/// The link between graph metrics and backend cost: instruction and
/// especially phi counts are what LLVM's verifier and instruction
/// selection actually pay for, so interface-shrinking passes should
/// move these, not just the graph numbers.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct EmittedIrStats {
    pub functions: usize,
    pub basic_blocks: usize,
    pub instructions: usize,
    pub phis: usize,
}

/// Rust-heap bytes at pipeline boundaries, filled by the DRIVER from
/// the [`heap`] counters (all zero when its binary does not install the
/// counting allocator). LLVM's C++ heap and the frontend subprocesses
/// are invisible to these numbers: the gap to peak RSS is them.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct HeapUsage {
    /// Live after frontend + parse: the llvm-ir AST.
    pub after_parse_bytes: usize,
    /// Live when the census ran: the graph, the AST already dropped.
    /// Vec::truncate keeps capacity, so a post-optimise census reports
    /// the same live bytes; the byte budget shows the logical shrink.
    pub live_at_census_bytes: usize,
    /// Process-lifetime peak so far; construction is where the AST and
    /// the graph coexist, so this usually dates from there.
    pub peak_bytes: usize,
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
    /// Value-operand fan-out samples, concatenated across functions.
    /// Ids are function-local since the per-function split, so only the
    /// counts carry meaning here (percentiles, maxima).
    pub fanout: Vec<u32>,
    /// Highest-fanout values module-wide, attributed by function name
    /// because the ValueId alone no longer identifies a value.
    pub top_fanout: Vec<(String, ValueId, &'static str, u32)>,
    /// Project values with zero uses: output slots exported by a
    /// construct that nothing consumes (dead-node elimination fodder).
    pub dead_projections: u64,
    /// Total value references (operand fields, span entries, region
    /// results): the sum of the fan-out array, i.e. the multiplier on
    /// any change to operand width.
    pub value_references: u64,
    pub memory_budget: MemoryBudget,
    pub span_composition: SpanComposition,

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

/// One aggregated graph-shape row per module: the census part of the
/// compile report (see [`CompileReportJson`]). Owned, so it outlives
/// the census it was derived from -- the full `ModuleCensus` holds
/// per-value sample vectors that must NOT stay alive across the pass
/// pipeline, or they poison every later heap measurement.
#[derive(Debug, Clone, Serialize)]
pub struct ModuleSummaryRow {
    pub module: String,
    pub functions: usize,
    pub values: u64,
    pub live_values: u64,
    pub regions: u64,
    pub thetas: u64,
    pub gammas: u64,
    pub promotion_candidates: u64,
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
    /// Region structs plus the region-span pool; the regions' blocks are
    /// inside `bytes_value_pool` and reported separately below, so the
    /// byte fields stay additive.
    pub bytes_regions: usize,
    /// The regions' interface/node blocks: a subset of
    /// `bytes_value_pool`, never added to a total.
    pub bytes_region_blocks: usize,
    pub bytes_match_arms: usize,
    pub span_gamma_inputs: usize,
    pub span_region_params: usize,
    pub span_region_results: usize,
    pub span_region_nodes: usize,
    pub span_loop_vars: usize,
    pub span_call_args: usize,
    pub span_ptr_offset_indices: usize,
    pub span_unaccounted: usize,
}

/// Exhaustive kind naming: a new `ValueKind` variant fails to compile
/// here until the census decides how to count it.
pub fn kind_name(kind: &ValueKind) -> &'static str {
    match kind {
        ValueKind::Const(_) => "Const",
        ValueKind::StateMerge { .. } => "StateMerge",
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
        ValueKind::Theta { .. } => "Theta",
        ValueKind::Gamma { .. } => "Gamma",
        ValueKind::Call { .. } => "Call",
        ValueKind::CallIndirect { .. } => "CallIndirect",
        ValueKind::Project { .. } => "Project",
        ValueKind::RegionParam { .. } => "RegionParam",
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
    ThetaBody { theta: ValueId },
    GammaArm { gamma: ValueId },
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
    graph: &'m FunctionGraph,
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
            for &node in self.graph.region_nodes(region_id) {
                match self.graph.get_value_kind(node) {
                    ValueKind::Gamma { regions, .. } => {
                        for &arm in self.graph.region_pool.get(*regions) {
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
            if let Some(position) = self
                .graph
                .region_params(region)
                .iter()
                .position(|&param| param == value)
            {
                break 'walk Some(position);
            }
            if let ValueKind::Project { call, index } = self.graph.get_value_kind(value)
                && let ValueKind::Gamma {
                    inputs, regions, ..
                } = self.graph.get_value_kind(*call)
            {
                let arms = self.graph.region_pool.get(*regions);
                let Some(inner_position) = self.gamma_slot_passthrough(arms, *index as usize)
                else {
                    break 'walk None;
                };
                let inputs = self.graph.value_pool.get(*inputs);
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
            let results = self.graph.region_results(arm);
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
            let ValueKind::Project { call, index } = self.graph.get_value_kind(value) else {
                break 'walk false;
            };
            let ValueKind::Gamma {
                inputs, regions, ..
            } = self.graph.get_value_kind(*call)
            else {
                break 'walk false;
            };
            let inputs = self.graph.value_pool.get(*inputs);
            let Some(position) = inputs
                .iter()
                .position(|&input| self.traces_to(input, target))
            else {
                break 'walk false;
            };
            self.graph
                .region_pool
                .get(*regions)
                .iter()
                .all(|&arm_region| {
                    let (Some(&arm_target), Some(&arm_result)) = (
                        self.graph.region_params(arm_region).get(position),
                        self.graph.region_results(arm_region).get(*index as usize),
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
        let kind = self.graph.get_value_kind(value);
        let result = if is_const_family(kind) {
            true
        } else {
            match owner_region(&self.owner, value) {
                Some(region) if !info.set.contains(&region) => true,
                None => false,
                Some(region) => {
                    if is_pure_compute(kind) {
                        let mut operands = Vec::new();
                        self.graph
                            .for_each_value_operand(value, |op| operands.push(op));
                        operands
                            .into_iter()
                            .all(|op| self.invariant_in(info, memo, op))
                    } else if let ValueKind::RegionParam { index, .. } = kind {
                        let index = *index as usize;
                        match info.roles.get(&region) {
                            Some(RegionRole::GammaArm { gamma }) => {
                                let ValueKind::Gamma { inputs, .. } =
                                    self.graph.get_value_kind(*gamma)
                                else {
                                    return false;
                                };
                                let inputs = self.graph.value_pool.get(*inputs);
                                index < inputs.len() && self.invariant_in(info, memo, inputs[index])
                            }
                            Some(RegionRole::ThetaBody { theta }) => {
                                let ValueKind::Theta { loop_vars, .. } =
                                    self.graph.get_value_kind(*theta)
                                else {
                                    return false;
                                };
                                let results = self.graph.region_results(RegionId(region));
                                let loop_vars = self.graph.value_pool.get(*loop_vars);
                                index < results.len()
                                    && index < loop_vars.len()
                                    && self.traces_to(results[index], value)
                                    && self.invariant_in(info, memo, loop_vars[index])
                            }
                            None => false,
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
            for &node in self.graph.region_nodes(region_id) {
                match self.graph.get_value_kind(node) {
                    ValueKind::Load { addr, .. }
                    | ValueKind::Store { addr, .. }
                    | ValueKind::AtomicLoad { addr, .. }
                    | ValueKind::AtomicStore { addr, .. }
                    | ValueKind::AtomicReadModifyWrite { addr, .. }
                    | ValueKind::CompareAndSwap { addr, .. } => {
                        mem_ops += 1;
                        let address = resolved
                            .entry(*addr)
                            .or_insert_with(|| self.graph.resolve_address(&self.m.tables, *addr));
                        bases.insert(address.base);
                    }
                    ValueKind::Fence { .. } => mem_ops += 1,
                    ValueKind::Call { fn_id, .. } => {
                        fc.calls_in_thetas += 1;
                        if effects_are_readonly(&self.m.tables.get_function(*fn_id).attrs) {
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
            for &node in self.graph.region_nodes(region_id) {
                let kind = self.graph.get_value_kind(node);
                match kind {
                    ValueKind::Load { addr, .. } | ValueKind::Store { addr, .. } => {
                        let external = is_const_family(self.graph.get_value_kind(*addr))
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
                    self.graph
                        .for_each_value_operand(node, |op| operands.push(op));
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
            for &store in self.graph.region_nodes(store_region) {
                let (cell_addr, cell_volatile) = match self.graph.get_value_kind(store) {
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
                    for &node in self.graph.region_nodes(region_id) {
                        if node == store {
                            continue;
                        }
                        match self.graph.get_value_kind(node) {
                            ValueKind::Call { fn_id, .. } => {
                                if !effects_are_readonly(&self.m.tables.get_function(*fn_id).attrs)
                                {
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
        let mut cross_region_seen: FxHashMap<Value, u32> = FxHashMap::default();

        // Visited guard: a function's Lambda node sits INSIDE its own
        // region, so the walk would otherwise revisit the root forever.
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        seen.insert(root.0);
        let mut stack: Vec<(RegionId, u32, u32)> = vec![(root, 0, 0)];
        while let Some((region_id, depth, loop_depth)) = stack.pop() {
            let region_nodes = self.graph.region_nodes(region_id);
            fc.regions += 1;
            fc.region_node_counts.push(region_nodes.len() as u32);
            fc.values += (region_nodes.len() + self.graph.region_params(region_id).len()) as u64;
            fc.max_depth = fc.max_depth.max(depth);
            fc.max_loop_depth = fc.max_loop_depth.max(loop_depth);

            let mut in_region_seen: FxHashSet<Value> = FxHashSet::default();
            for &node in region_nodes {
                let value_kind = self.graph.get_value_kind(node);
                match value_kind {
                    ValueKind::Gamma { regions, .. } => {
                        let arms = self.graph.region_pool.get(*regions);
                        fc.gammas += 1;
                        fc.gamma_arities.push(arms.len() as u32);
                        let result_count = arms
                            .first()
                            .map(|&arm| self.graph.region_results(arm).len())
                            .unwrap_or(0);
                        fc.gamma_outputs += result_count as u64;
                        for slot in 0..result_count {
                            if self.gamma_slot_passthrough(arms, slot).is_some() {
                                fc.gamma_passthrough += 1;
                            }
                        }
                        for &arm in arms {
                            for &result in self.graph.region_results(arm) {
                                fc.gamma_result_entries += 1;
                                if matches!(
                                    self.graph.get_value_kind(result),
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
                            .push(self.graph.value_pool.get(*loop_vars).len() as u32);
                        let results = self.graph.region_results(*body);
                        fc.theta_outputs += results.len() as u64;
                        for (slot, &result) in results.iter().enumerate() {
                            if let Some(&param) = self.graph.region_params(*body).get(slot)
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
                    ValueKind::Match { .. } => fc.matches += 1,
                    ValueKind::Project { .. } => fc.projections += 1,
                    _ => {}
                }

                // CNE dry run: identical pure/constant values. Spans are
                // compared by id, so structurally equal nodes with
                // separately pooled spans undercount; conservative.
                if is_pure_compute(value_kind) || is_const_family(value_kind) {
                    let ty = self.graph.get_value_type(node);
                    let value = Value {
                        ty: *ty,
                        kind: *value_kind,
                    };
                    if !in_region_seen.insert(value.clone()) {
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
                    value_kind,
                    ValueKind::Unary { .. }
                        | ValueKind::Binary { .. }
                        | ValueKind::ICmp { .. }
                        | ValueKind::FCmp { .. }
                        | ValueKind::Cast { .. }
                        | ValueKind::Ternary { .. }
                ) {
                    let mut all_const = true;
                    self.graph.for_each_value_operand(node, |op| {
                        all_const &= matches!(
                            self.graph.get_value_kind(op),
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
    let mut census = ModuleCensus {
        mod_name: m.mod_name.clone(),
        interned_types: m.tables.types.interned_len(),
        interned_signatures: m.tables.signatures.len(),
        interned_constants: m.tables.constants.len(),
        globals: m.tables.globals.len(),
        ..Default::default()
    };

    // Everything below aggregates over the per-function graphs: pool
    // lengths and byte budgets sum, fan-out samples concatenate, and the
    // interner counts above stay module-scoped.
    let mut spans = SpanComposition::default();
    let mut budget = MemoryBudget::default();
    let mut top_candidates: Vec<(String, ValueId, &'static str, u32)> = Vec::new();

    for (function, graph) in m.tables.functions.iter().zip(&m.graphs) {
        let Some(graph) = graph else { continue };
        census.total_values += graph.value_kinds.len() as u64;
        census.value_pool_len += graph.value_pool.len();
        census.region_pool_len += graph.region_pool.len();
        census.u32_pool_len += graph.u32_pool.len();
        census.match_arm_pool_len += graph.match_arm_pool.len();

        // Kind histogram + projection parents.
        for value in &graph.value_kinds {
            *census.kind_counts.entry(kind_name(value)).or_insert(0) += 1;
            if let ValueKind::Project { call, .. } = value {
                let parent = kind_name(graph.get_value_kind(*call));
                *census.projections_by_parent.entry(parent).or_insert(0) += 1;
            }
        }

        // Fan-out: value-operand uses plus region results.
        let mut fanout = vec![0u32; graph.value_kinds.len()];
        for (index, _value) in graph.value_kinds.iter().enumerate() {
            graph.for_each_value_operand(ValueId(index as u32), |op| {
                fanout[op.0 as usize] += 1;
            });
        }
        for region_index in 0..graph.regions.len() {
            for &result in graph.region_results(RegionId(region_index as u32)) {
                fanout[result.0 as usize] += 1;
            }
        }
        census.dead_projections += graph
            .value_kinds
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                matches!(value, ValueKind::Project { .. }) && fanout[*index] == 0
            })
            .count() as u64;
        census.value_references += fanout.iter().map(|&uses| uses as u64).sum::<u64>();

        let mut ranked: Vec<usize> = (0..graph.value_kinds.len()).collect();
        ranked.sort_unstable_by_key(|&index| std::cmp::Reverse(fanout[index]));
        top_candidates.extend(
            ranked
                .into_iter()
                .take(10)
                .filter(|&index| fanout[index] > 0)
                .map(|index| {
                    (
                        function.name.clone(),
                        ValueId(index as u32),
                        kind_name(&graph.value_kinds[index]),
                        fanout[index],
                    )
                }),
        );

        // Byte budget of the backing arrays. Kinds and types are split
        // vectors since the SoA change, so both are counted explicitly.
        budget.values_bytes += graph.value_kinds.len() * size_of::<ValueKind>()
            + graph.value_types.len() * size_of::<TypeRef>();
        budget.value_pool_bytes += graph.value_pool.len() * size_of::<ValueId>();
        budget.region_structs_bytes +=
            graph.regions.len() * size_of::<crate::rvsdg::region::Region>();
        // Region interface and node blocks live in the value pool, so
        // their bytes are counted in value_pool_bytes; this field breaks
        // out the block subtotal (a SUBSET, never added to a total).
        budget.region_blocks_bytes += graph
            .regions
            .iter()
            .map(|region| {
                (region.params_len as usize
                    + region.results_len as usize
                    + region.nodes_len as usize)
                    * size_of::<ValueId>()
            })
            .sum::<usize>();
        budget.region_pool_bytes += graph.region_pool.len() * size_of::<RegionId>();
        budget.u32_pool_bytes += graph.u32_pool.len() * size_of::<u32>();
        budget.match_arm_pool_bytes +=
            graph.match_arm_pool.len() * size_of::<crate::rvsdg::MatchArm>();

        // value_pool composition: which span kinds fill it.
        for value in &graph.value_kinds {
            match value {
                ValueKind::Gamma { inputs, .. } => spans.gamma_inputs += inputs.len as usize,
                ValueKind::Theta { loop_vars, .. } => {
                    spans.theta_loop_vars += loop_vars.len as usize
                }
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
        for region in &graph.regions {
            spans.region_params += region.params_len as usize + region.state_params_len as usize;
            spans.region_results += region.results_len as usize + region.state_results_len as usize;
            spans.region_nodes += region.nodes_len as usize;
        }

        // Liveness: region membership seeds a value-operand closure. The gap
        // to total_values is the husk fraction passes leave behind.
        let mut live = vec![false; graph.value_kinds.len()];
        let mut worklist: Vec<ValueId> = Vec::new();
        let mark = |live: &mut Vec<bool>, worklist: &mut Vec<ValueId>, value: ValueId| {
            if !live[value.0 as usize] {
                live[value.0 as usize] = true;
                worklist.push(value);
            }
        };
        for region_index in 0..graph.regions.len() {
            let region_id = RegionId(region_index as u32);
            for &state_param in graph.region_state_params(region_id) {
                mark(&mut live, &mut worklist, state_param);
            }
            for &state_result in graph.region_state_results(region_id) {
                mark(&mut live, &mut worklist, state_result);
            }
            for &param in graph.region_params(region_id) {
                mark(&mut live, &mut worklist, param);
            }
            for &node in graph.region_nodes(region_id) {
                mark(&mut live, &mut worklist, node);
            }
            for &result in graph.region_results(region_id) {
                mark(&mut live, &mut worklist, result);
            }
        }
        let mut operands = Vec::new();
        while let Some(value) = worklist.pop() {
            operands.clear();
            graph.for_each_value_operand(value, |op| operands.push(op));
            for &op in &operands {
                mark(&mut live, &mut worklist, op);
            }
        }
        census.live_values += live.iter().filter(|&&l| l).count() as u64;

        // Fan-out samples feed the module-wide percentiles; ids are
        // function-local, so only the counts carry meaning here.
        census.fanout.extend_from_slice(&fanout);

        // Region-tree walk: one collector per graph, since the memo
        // caches key on function-local ids. Region 0 is the body root by
        // the graph constructor's convention.
        let mut ownership_errs = Vec::new();
        let collector = Collector {
            owner: graph.build_value_ownership(&mut ownership_errs),
            graph,
            m,
            traces_memo: Default::default(),
            reduces_memo: Default::default(),
            passthrough_memo: Default::default(),
        };
        census
            .functions
            .push(collector.function_census(function.name.clone(), RegionId(0)));
    }

    spans.unaccounted = census.value_pool_len.saturating_sub(
        spans.gamma_inputs
            + spans.region_params
            + spans.region_results
            + spans.region_nodes
            + spans.theta_loop_vars
            + spans.call_args
            + spans.ptr_offset_indices
            + spans.shuffle_masks,
    );
    census.span_composition = spans;
    census.memory_budget = budget;

    top_candidates.sort_unstable_by_key(|&(_, _, _, uses)| std::cmp::Reverse(uses));
    top_candidates.truncate(10);
    census.top_fanout = top_candidates;

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
    pub fn summary_row(&self) -> ModuleSummaryRow {
        let sum = |f: fn(&FunctionCensus) -> u64| self.functions.iter().map(f).sum::<u64>();
        ModuleSummaryRow {
            module: self.mod_name.clone(),
            functions: self.functions.len(),
            values: self.total_values,
            live_values: self.live_values,
            regions: sum(|f| f.regions),
            thetas: sum(|f| f.thetas),
            gammas: sum(|f| f.gammas),
            promotion_candidates: sum(|f| f.promotion_candidates),
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
                + self.memory_budget.region_pool_bytes,
            bytes_region_blocks: self.memory_budget.region_blocks_bytes,
            bytes_match_arms: self.memory_budget.match_arm_pool_bytes,
            span_gamma_inputs: self.span_composition.gamma_inputs,
            span_region_params: self.span_composition.region_params,
            span_region_results: self.span_composition.region_results,
            span_region_nodes: self.span_composition.region_nodes,
            span_loop_vars: self.span_composition.theta_loop_vars,
            span_call_args: self.span_composition.call_args,
            span_ptr_offset_indices: self.span_composition.ptr_offset_indices,
            span_unaccounted: self.span_composition.unaccounted,
        }
    }

    /// Human summary. The caller supplies the sink (a stderr lock from
    /// the compiler's --stats path; a Vec in tests).
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
        for (func, value, kind, uses) in &self.top_fanout {
            writeln!(out, "  {func} {value:?} {kind:<14} {uses} uses")?;
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
            summary.promotion_candidates,
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
            "-- memory budget -- values {}KB, value_pool {}KB (of which region blocks {}KB), regions {}KB (structs {} + span pool {}), match arms {}KB",
            budget.values_bytes / 1024,
            budget.value_pool_bytes / 1024,
            budget.region_blocks_bytes / 1024,
            (budget.region_structs_bytes + budget.region_pool_bytes) / 1024,
            budget.region_structs_bytes / 1024,
            budget.region_pool_bytes / 1024,
            budget.match_arm_pool_bytes / 1024,
        )?;
        let spans = &self.span_composition;
        writeln!(
            out,
            "-- value_pool spans -- gamma inputs {}, region params {}, region results {}, region nodes {}, loop vars {}, call args {}, ptr offset indices {}, shuffle masks {}, unaccounted {}",
            spans.gamma_inputs,
            spans.region_params,
            spans.region_results,
            spans.region_nodes,
            spans.theta_loop_vars,
            spans.call_args,
            spans.ptr_offset_indices,
            spans.shuffle_masks,
            spans.unaccounted,
        )?;
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
        types::{I32, PtrType, TypeRef},
    };

    fn ptr_ty(rvsdg: &mut RVSDGMod) -> TypeRef {
        let id = rvsdg.tables.types.intern_ptr(PtrType {
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
            .define_fn(func_id, |rb| {
                let one = rb.const_i32(1);
                let alloc = rb.alloca(I32, one, ptr, None);
                let zero = rb.const_i32(0);
                rb.store(alloc, zero, None, false);
                let i = rb.const_i32(0);
                rb.theta(&[i], |rb| {
                    let loop_i = rb.param(0);
                    let cur = rb.load(alloc, I32, None, false);
                    let ten = rb.const_i32(10);
                    let next_val = rb.binary(BinaryOp::Add, ArithFlags::default(), cur, ten, I32);
                    rb.store(alloc, next_val, None, false);
                    let one = rb.const_i32(1);
                    let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                    let five = rb.const_i32(5);
                    let cond = rb.icmp(ICmpPred::SignedLt, next_i, five);
                    Ok(LoopResult {
                        condition: cond,
                        next_vars: vec![next_i],
                    })
                })?;
                let final_val = rb.load(alloc, I32, None, false);
                Ok(vec![final_val])
            })
            .unwrap();

        let census = collect(&rvsdg);
        // Accounting invariants the reporting relies on: every pool entry
        // lands in exactly one span bucket (unaccounted means orphaned
        // spans, and a fresh module has none), and the regions' blocks
        // are a subset of the pool they live in. Pinned here so a future
        // double-count fails instead of being clamped by saturating_sub.
        assert_eq!(census.span_composition.unaccounted, 0);
        assert!(census.memory_budget.region_blocks_bytes <= census.memory_budget.value_pool_bytes);

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
            .define_fn(func_id, |rb| {
                let i = rb.const_i32(0);
                rb.theta(&[i], |rb| {
                    let loop_i = rb.param(0);
                    let one = rb.const_i32(1);
                    let alloc = rb.alloca(I32, one, ptr, None);
                    rb.store(alloc, loop_i, None, false);
                    let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                    let five = rb.const_i32(5);
                    let cond = rb.icmp(ICmpPred::SignedLt, next_i, five);
                    Ok(LoopResult {
                        condition: cond,
                        next_vars: vec![next_i],
                    })
                })?;
                let zero = rb.const_i32(0);
                Ok(vec![zero])
            })
            .unwrap();

        let census = collect(&rvsdg);
        let f = &census.functions[0];
        assert_eq!(f.promotion_candidates, 0, "census: {f:?}");
        assert_eq!(f.bail_varying, 1, "census: {f:?}");
    }
}
