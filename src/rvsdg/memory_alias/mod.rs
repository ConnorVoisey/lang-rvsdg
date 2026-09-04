use smallvec::SmallVec;

use tracing::instrument;

use crate::rvsdg::{
    FuncId, GlobalId, RVSDGMod, ValueId, ValueKind, func::MemReadWrite,
    function_graph::FunctionGraph, memory_alias::origin::MemoryOrigin, module_tables::ModuleTables,
    types::TypeRef,
};

pub mod classify;
pub mod classing;
pub mod origin;
pub mod resolve;

/// Info about the values memory, created at construction of function,
/// used at the end of function resolution.
#[derive(Debug, Default)]
pub struct MemoryFactScratch {
    /// Values that are "escaping": stored as data, cast to integer, or
    /// returned. Call arguments are NOT recorded here -- they are
    /// judged from CallFacts, so a non-retaining callee costs nothing.
    /// For store `ptr %p, ptr %slot` the stores own id goes into access_events,
    /// whereas %p's id goes here.
    /// `store i32 7, ptr %slot` records nothing since publishing a number leaks no address.
    pub(crate) escape_events: Vec<ValueId>,

    /// Memory operation values: loads, stores, atomics, memory intrinsics
    pub(crate) access_events: Vec<ValueId>,

    /// (join value, extra input) pairs
    pub(crate) join_events: Vec<(ValueId, ValueId)>,

    /// Call and call indirect value ids, resolved into [`CallFact`] at build finish
    pub(crate) call_sites: Vec<ValueId>,

    /// The function's returned values, recorded by define_fn. Kept
    /// apart from escape_events because a returned CallResult is NOT an
    /// anonymous escape: the barrier's Fresh rule needs to see that the
    /// only escaping position was the return itself.
    pub(crate) returns: Vec<ValueId>,

    /// Alloca node ids in construction order (ascending), so the
    /// classing candidate list is born sorted and no later pass has to
    /// re-walk the values to find them.
    pub(crate) allocas: Vec<ValueId>,

    /// The origin resolver's work buffers (deferred list, Tarjan walk
    /// state), recycled like the event lists. Every field is
    /// re-initialised at use, so clear() does not touch it.
    pub(crate) resolve: resolve::ResolveScratch,
}

impl MemoryFactScratch {
    /// Empty the event lists keeping their capacity, so a recycled
    /// scratch's buffers stay warm across functions. resolve_facts
    /// calls this once after all classification passes have read the
    /// events; it also covers the error path, where no resolution runs.
    pub(crate) fn clear(&mut self) {
        self.escape_events.clear();
        self.access_events.clear();
        self.join_events.clear();
        self.call_sites.clear();
        self.returns.clear();
        self.allocas.clear();
    }
}

impl FunctionGraph {
    /// Record a returned value. Classified like a hard escape except
    /// for CallResult origins (see the `returns` field).
    #[inline]
    pub(crate) fn record_return_event(&mut self, value: ValueId) {
        self.mem_facts.returns.push(value);
    }
}

// Event recording, called from the builder methods. Events are raw ids
// because the tags they classify against are not final until
// finish_building (the one rule).
impl FunctionGraph {
    /// Record a value in a hard-escaping position. The gate lives HERE
    /// so call sites stay unconditional: publishing a number leaks no
    /// address, and a pointer packed inside an aggregate or vector
    /// already escaped at its insert site. The one non-pointer that
    /// smuggles addresses is an aggregate POOL CONSTANT -- `{ &f }` has
    /// no insert site -- so those pass the gate and classification
    /// walks their payloads.
    #[inline]
    pub(crate) fn record_escape_event(&mut self, value: ValueId) {
        let index = value.0 as usize;
        let is_pointer = matches!(self.value_types[index], TypeRef::Ptr(_));
        if is_pointer || matches!(self.value_kinds[index], ValueKind::ConstPoolRef(_)) {
            self.mem_facts.escape_events.push(value);
        }
    }

    /// Record a memory op node; its addr operand's origin decides at
    /// resolution which class the op belongs to.
    #[inline]
    pub(crate) fn record_access_event(&mut self, op: ValueId) {
        self.mem_facts.access_events.push(op);
    }

    /// Record an extra input of a multi-input pointer join (region
    /// param recirculation, construct output projections, select).
    #[inline]
    pub(crate) fn record_join_event(&mut self, join: ValueId, input: ValueId) {
        self.mem_facts.join_events.push((join, input));
    }

    /// Record a Call/CallIndirect node for CallFact resolution.
    #[inline]
    pub(crate) fn record_call_site(&mut self, op: ValueId) {
        self.mem_facts.call_sites.push(op);
    }

    /// Record an alloca node: the classing candidate inventory.
    #[inline]
    pub(crate) fn record_alloca_event(&mut self, node: ValueId) {
        self.mem_facts.allocas.push(node);
    }
}

impl FunctionGraph {
    /// The end-of-construction resolution, called by define_fn once the
    /// body is built: compress origins (Tarjan over the Derived/join
    /// edges), then classify every recorded event through the final
    /// tags. Origin resolution can APPEND widening escapes, so it runs
    /// before the escape pass reads. The classification passes only
    /// read the event lists; one clear at the end keeps every buffer's
    /// capacity warm for the next function on this scratch.
    #[instrument(skip_all, fields(values = self.value_kinds.len()))]
    pub(crate) fn resolve_facts(&mut self, tables: &ModuleTables) -> FunctionFacts {
        let mut facts = FunctionFacts::empty();
        self.resolve_origins();
        self.classify_escape_events(tables, &mut facts);
        self.classify_access_events(&mut facts);
        self.classify_calls_and_returns(tables, &mut facts);
        facts.allocas.extend_from_slice(&self.mem_facts.allocas);
        facts.finalize();
        self.mem_facts.clear();
        facts
    }
}

/// One call site, not one function. Resolved at finished building.
/// Callee is either Some(FuncId) when we know the symbol or None if it is passed as a function
/// pointer and only known at runtime.
///
/// - Known callee: arg_provenance is indexed by the callee's DECLARED
///   parameter positions (None entries for non-pointer parameters).
///   Pointer arguments beyond the declared list (varargs) or at
///   signature-mismatched sites are not translated: they resolve as
///   hard escapes instead.
///
/// - callee None (indirect): arg_provenance is positional over the
///   actual arguments. The fold applies the external node (TOP
///   captures), so every named origin in it escapes; positional is
///   equivalent to any other indexing there.
#[derive(Debug, Clone)]
pub struct CallFact {
    pub callee: Option<FuncId>,

    /// provenance per argument.
    pub arg_provenance: SmallVec<[MemoryOrigin; 6]>,
}

/// What a function's pointer-typed result traces to, as far as the
/// function alone can tell. The barrier finishes CallResult cases
/// through the callee's return provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalReturn {
    /// No usable fact (also: no pointer-typed result).
    Unknown,
    /// Returns (a pointer derived from) the index-th parameter.
    Param(u8),
    /// Returns the result of call site `calls[index]`.
    CallResult(u32),
}

/// End of construction facts about a function. A pre-DNE snapshot:
/// dead escapes and accesses leave conservative residue, accepted.
/// Built by `empty()`, written by hand -- "observed nothing" is the
/// correct empty for FACTS, while the summary-side bitfield newtypes
/// deliberately have no Default so TOP seeding stays explicit.
#[derive(Debug)]
pub struct FunctionFacts {
    /// Origins whose address hard-escapes: alloca node ids, and later
    /// Fresh-call ids. Sorted by id and deduped at classification.
    pub escaped_origins: SmallVec<[ValueId; 4]>,

    /// Every alloca node, ascending by id: the classing candidates,
    /// recorded at construction so classing never walks the values.
    pub allocas: SmallVec<[ValueId; 2]>,
    /// Allocas with a volatile or atomic access, sorted and deduped.
    /// Folding is whole-function: one op has one chain membership, so
    /// pinning only the offending site would leave two accesses to the
    /// same slot on different chains with no edge between them.
    pub folded_volatile_atomic: SmallVec<[ValueId; 2]>,
    /// Allocas a multi-address intrinsic ties to another class, sorted
    /// and deduped: the op carries a single state operand and cannot be
    /// a member of two chains, so every alloca it touches folds.
    pub folded_multi_address: SmallVec<[ValueId; 2]>,
    /// Alloca origins handed to any call, sorted and deduped: with no
    /// callee knowledge, every call retains what it is handed. The
    /// summaries tier refines this through captured-param facts.
    pub call_retained: SmallVec<[ValueId; 2]>,

    /// Own accesses through memory a caller might name but no finer
    /// fact covers (Unknown provenance; untracked globals fold in at
    /// the barrier).
    pub local_other: MemReadWrite,
    /// Own accesses through each pointer parameter.
    pub local_param_effects: ParamEffects,
    /// Own parameters that hard-escape: the seed captures grow from.
    pub local_captured: CapturedParams,
    /// Globals the function's own ops may read / may write.
    pub global_access: SmallVec<[(GlobalId, MemReadWrite); 4]>,
    /// Globals whose ADDRESS this function leaks.
    pub address_taken_global: SmallVec<[GlobalId; 2]>,
    /// Functions whose ADDRESS this function leaks (function pointers
    /// stashed anywhere make the function externally callable).
    pub address_taken_functions: SmallVec<[FuncId; 2]>,
    /// The function's call sites, resolved and id-free at the boundary.
    pub calls: Vec<CallFact>,
    /// See [`LocalReturn`].
    pub local_return: LocalReturn,
    /// Accesses whose addr origin is a CallResult: classifiable only
    /// once the callee's return provenance is known, so they wait for
    /// the barrier as (call index, effect).
    pub deferred_accesses: SmallVec<[(u32, MemReadWrite); 2]>,
    /// Escaping values whose origin is a CallResult, as call indices.
    pub deferred_escapes: SmallVec<[u32; 2]>,
}

impl FunctionFacts {
    /// Bring the facts through a DNE compaction: live ids remap, dead
    /// entries drop (a dead origin has no surviving accesses to
    /// conflict with). The call-INDEX lists (deferred_*,
    /// LocalReturn::CallResult) are id-free and stay valid; the
    /// mapper is monotonic, so sorted lists stay sorted.
    pub(crate) fn remap_ids(&mut self, value_mapper: &[u32]) {
        remap_value_id_list(&mut self.escaped_origins, value_mapper);
        remap_value_id_list(&mut self.allocas, value_mapper);
        remap_value_id_list(&mut self.folded_volatile_atomic, value_mapper);
        remap_value_id_list(&mut self.folded_multi_address, value_mapper);
        remap_value_id_list(&mut self.call_retained, value_mapper);
        for call in &mut self.calls {
            for origin in &mut call.arg_provenance {
                *origin = origin.remap(value_mapper);
            }
        }
    }

    pub fn empty() -> Self {
        FunctionFacts {
            escaped_origins: SmallVec::new(),
            allocas: SmallVec::new(),
            folded_volatile_atomic: SmallVec::new(),
            folded_multi_address: SmallVec::new(),
            call_retained: SmallVec::new(),
            local_other: MemReadWrite::None,
            local_param_effects: ParamEffects::EMPTY,
            local_captured: CapturedParams::EMPTY,
            global_access: SmallVec::new(),
            address_taken_global: SmallVec::new(),
            address_taken_functions: SmallVec::new(),
            calls: Vec::new(),
            local_return: LocalReturn::Unknown,
            deferred_accesses: SmallVec::new(),
            deferred_escapes: SmallVec::new(),
        }
    }
}

/// Remap one ValueId list through a DNE mapper: dead entries drop,
/// live entries take their new id. The mapper is monotonic, so a
/// sorted list stays sorted.
fn remap_value_id_list(
    list: &mut SmallVec<impl smallvec::Array<Item = ValueId>>,
    value_mapper: &[u32],
) {
    list.retain(|value| value_mapper[value.0 as usize] != u32::MAX);
    for value in list.iter_mut() {
        *value = ValueId(value_mapper[value.0 as usize]);
    }
}

/// A global is tracked if it has internal linkage and its address is never taken.
/// Dense index over tracked globals. Doubles as a bit position within the arenas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackedGlobalId(pub u32);

#[derive(Debug)]
pub struct GlobalSetArena {
    words: Vec<u64>,
    stride: u32,
}
impl GlobalSetArena {
    pub fn contains(&self, row: usize, g: TrackedGlobalId) -> bool {
        todo!()
    }
    fn union_rows(&mut self, into: usize, from: usize) -> bool {
        todo!()
    }
}

/// What calling this function may do, transitively. A few Copy bytes;
/// the global rows live in FnSummaries' arenas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FnSummary {
    /// Per pointer-parameter may-read/may-write, declared-position
    /// indexed. Per-param rather than one argmem fact because
    /// translation must not smear one argument's effect onto
    /// another's chain.
    pub param_effects: ParamEffects,
    /// Effects on memory the caller can name but no finer fact covers:
    /// unknown-provenance accesses and untracked globals. When seeding
    /// from LLVM declarations, errnomem and inaccessiblemem fold in
    /// here too -- the optimizer keeps ONE "somewhere out there"
    /// location; LLVM's four-location memory(...) attribute lives only
    /// in FnAttrs for round-trip.
    pub other: MemReadWrite,
    /// May transitively perform io.
    pub does_io: bool,
    /// Pointer parameters that may be RETAINED past the call (stored,
    /// returned, still reachable afterwards). The CLEAR bit is the
    /// valuable fact: passing a private alloca to a non-retaining
    /// parameter is one op on its chain, not an escape.
    pub captured_params: CapturedParams,
}

/// Fixed width bitset,
/// each bit is for a parameter n: 0 for not captured, 1 for captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapturedParams(u32);
impl CapturedParams {
    pub const CAPACITY: u32 = 32;
    /// seed for default
    pub const EMPTY: Self = CapturedParams(0);
    /// All retained for conservative worse case: unknown callees, external node
    pub const TOP: Self = CapturedParams(u32::MAX);
    pub fn is_retained(self, param: u32) -> bool {
        if param < Self::CAPACITY {
            self.0 & (1 << param) != 0
        } else {
            true
        }
    }
    pub fn set_retained(&mut self, param: u32) {
        // Can only store Self::CAPACITY info about retaining params.
        // If this is frequently exceeded on real code, could move to a u64.
        if param < Self::CAPACITY {
            self.0 |= 1 << param;
        }
    }
}

/// Fixed width bitset, 2 bits per parameter.
/// Each params first bit is has read, second is has write
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamEffects(u64);
impl ParamEffects {
    pub const CAPACITY: u32 = 32;
    pub const EMPTY: Self = ParamEffects(0);
    /// All read and write for worse case
    pub const TOP: Self = ParamEffects(u64::MAX);
    pub fn get(self, param: u32) -> MemReadWrite {
        if param >= Self::CAPACITY {
            MemReadWrite::ReadAndWrite
        } else {
            MemReadWrite::from_bits((self.0 >> (param * 2)) as u8)
        }
    }

    /// JOIN, not overwrite: two accesses through one parameter
    /// (ReadOnly then WriteOnly) must yield ReadAndWrite.
    pub fn join(&mut self, param: u32, effect: MemReadWrite) {
        if param < Self::CAPACITY {
            self.0 |= (effect as u64) << (param * 2);
        }
    }
}

/// The lattice join everything above relies on: bitwise OR over the
/// {may-read, may-write} bit pair. Nothing in the fold may assign an
/// effect; it joins.
impl MemReadWrite {
    pub fn join(self, other: Self) -> Self {
        Self::from_bits(self as u8 | other as u8)
    }
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::None,
            0b01 => Self::ReadOnly,
            0b10 => Self::WriteOnly,
            _ => Self::ReadAndWrite,
        }
    }
}

/// FuncId-indexed; immutable once propagation completes, so every
/// later per-function pass reads it in parallel without coordination.
#[derive(Debug)]
pub struct FnSummaries {
    summaries: Vec<FnSummary>,
    read_globals: GlobalSetArena,
    write_globals: GlobalSetArena,
    /// Escapes discovered at the barrier: indexed by CALLER FuncId,
    /// holding the caller's Alloca ValueIds whose address was passed
    /// where the callee may retain it. Alloca origins ONLY -- a
    /// retained Param becomes the caller's own captured bit, a
    /// retained Global becomes address-taken (both in barrier stage
    /// 1). Consumed by classing (escaped_allocas UNION
    /// summary_escapes[f]) and dead after the final rebuild; never
    /// remapped.
    summary_escapes: Vec<SmallVec<[ValueId; 2]>>,
}

impl RVSDGMod {
    pub fn compute_fn_summaries(&self, facts: &[FunctionFacts]) -> FnSummaries {
        todo!()
    }
}

/// The stage-3 fold for one function: its facts plus its callees'
/// summaries. NO graph access -- everything needed is in the facts,
/// which is what keeps the barrier graph-free. Per CallFact and callee
/// parameter k, param_effects.get(k) translates through
/// arg_provenance[k]:
///
///   Alloca (still private)   -> nothing: invisible outside our frame
///   Param(j) forwarded       -> joined into our param_effects entry j
///   Global(g)                -> our (g, effect) pair; `other` if untracked
///   Unknown                  -> our `other`
///
/// Callee `other` and does_io join directly (they never depended on
/// what was passed); global rows union per arena. callee == None
/// folds the external node.
fn fold_summary(facts: &FunctionFacts, summaries: &FnSummaries) -> FnSummary {
    todo!()
}

#[cfg(test)]
mod test {
    use crate::rvsdg::{
        func::MemReadWrite,
        memory_alias::{CapturedParams, ParamEffects},
    };

    #[test]
    fn captured_params_basic() {
        let mut cap = CapturedParams::EMPTY;

        for i in 0..CapturedParams::CAPACITY {
            assert!(!cap.is_retained(i));
        }

        let retained = 5;
        cap.set_retained(retained);
        for i in 0..CapturedParams::CAPACITY {
            if i == retained {
                assert!(cap.is_retained(retained));
            } else {
                assert!(!cap.is_retained(i));
            }
        }
    }

    #[test]
    fn captured_params_retained() {
        let mut cap = CapturedParams::EMPTY;

        let retained = [0, 2, 3, 4, 5, 9, 31];
        for i in 0..CapturedParams::CAPACITY {
            assert!(!cap.is_retained(i));
        }

        for i in retained {
            cap.set_retained(i);
            assert!(cap.is_retained(i));
        }
        for i in 0..CapturedParams::CAPACITY {
            if retained.contains(&i) {
                assert!(cap.is_retained(i));
            } else {
                assert!(!cap.is_retained(i));
            }
        }

        // setting retained again should have no effect
        for i in retained {
            cap.set_retained(i);
            assert!(cap.is_retained(i));
        }
        for i in 0..CapturedParams::CAPACITY {
            if retained.contains(&i) {
                assert!(cap.is_retained(i));
            } else {
                assert!(!cap.is_retained(i));
            }
        }
    }

    #[test]
    fn captured_params_outside_bounds() {
        let mut cap = CapturedParams::EMPTY;
        let i = CapturedParams::CAPACITY;
        cap.set_retained(i);
        assert!(cap.is_retained(i));
    }

    #[test]
    fn param_effects_empty() {
        let param_effects = ParamEffects::EMPTY;

        for i in 0..ParamEffects::CAPACITY {
            let eff = param_effects.get(i);
            assert!(matches!(eff, MemReadWrite::None));
        }
    }

    #[test]
    fn param_effects_top() {
        let param_effects = ParamEffects::TOP;

        for i in 0..ParamEffects::CAPACITY {
            let eff = param_effects.get(i);
            assert!(matches!(eff, MemReadWrite::ReadAndWrite));
        }
    }

    #[test]
    fn param_effects_basic() {
        let mut param_effects = ParamEffects::EMPTY;

        let to_check = [0, 1, 5, 6, 31];
        for i in to_check {
            param_effects.join(i, MemReadWrite::ReadOnly);
            assert!(matches!(param_effects.get(i), MemReadWrite::ReadOnly));

            param_effects.join(i, MemReadWrite::ReadOnly);
            assert!(
                matches!(param_effects.get(i), MemReadWrite::ReadOnly),
                "joining with the same shouldn't change"
            );

            param_effects.join(i, MemReadWrite::WriteOnly);
            assert!(matches!(param_effects.get(i), MemReadWrite::ReadAndWrite));

            param_effects.join(i, MemReadWrite::None);
            assert!(
                matches!(param_effects.get(i), MemReadWrite::ReadAndWrite),
                "joining on None shouldn't change anything"
            );
        }
    }
}
