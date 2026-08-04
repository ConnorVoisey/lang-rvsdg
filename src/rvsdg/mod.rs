pub mod alias;
pub mod builder;
pub mod constant;
pub mod dump;
pub mod func;
pub mod function_graph;
pub mod global;
pub mod lower_to_llvm;
pub mod module_tables;
pub mod ops;
pub mod types;
pub mod value;
pub mod verify;

pub use constant::{ConstId, ConstIdPool, ConstIdsSpan, ConstantDef, ConstantKind, ConstantPool};
pub use global::{GlobalDef, GlobalInit, ThreadLocalMode};
pub use ops::{
    ArithFlags, AtomicRMWOp, BinaryOp, CastOp, FCmpPred, ICmpPred, IntrinsicOp, MemoryOrdering,
    UnaryOp,
};
pub use target_lexicon::Triple;
pub use value::{ConstValue, Value, ValueKind};

use crate::rvsdg::{function_graph::FunctionGraph, module_tables::ModuleTables};

#[derive(Debug)]
pub struct RVSDGMod {
    /// Target triple (e.g. x86_64-unknown-linux-gnu)
    pub target: Triple,
    pub mod_name: String,
    /// LLVM data layout string -- encodes pointer sizes, alignments, endianness
    /// for the target. Preserved verbatim for roundtripping through LLVM.
    pub data_layout: String,
    /// Module-level inline assembly (`module asm "..."` lines), preserved
    /// verbatim: it defines real symbols (e.g. hand-written context-switch
    /// routines) that the rest of the module references.
    pub module_asm: String,
    pub tables: ModuleTables,
    pub graphs: Vec<Option<FunctionGraph>>,
}

impl RVSDGMod {
    pub fn new(mod_name: String, target: Triple, data_layout: String) -> Self {
        Self {
            target,
            mod_name,
            data_layout,
            module_asm: String::new(),
            tables: ModuleTables::default(),
            graphs: Vec::default(),
        }
    }

    /// Create a module targeting the host platform with an empty data layout.
    pub fn new_host(mod_name: String) -> Self {
        Self::new(mod_name, Triple::host(), String::new())
    }

    pub fn get_graph(&self, func_id: FuncId) -> &Option<FunctionGraph> {
        &self.graphs[func_id.0 as usize]
    }
}

/// Primary handle into the IR. Indexes a function graph's value arrays
/// (ids are function-local since the per-function split).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Sentinel for "no value". Deliberately out of range so accidental
    /// use panics at the first indexed access instead of silently
    /// resolving to a real value.
    pub const INVALID: ValueId = ValueId(u32::MAX);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

impl RegionId {
    /// Sentinel for "no region". Same fail-fast rationale as
    /// [`ValueId::INVALID`].
    pub const INVALID: RegionId = RegionId(u32::MAX);
}

#[derive(Debug, Clone, Default)]
pub struct ValuePool(Vec<ValueId>);

impl ValuePool {
    pub fn push_slice(&mut self, values: &[ValueId]) -> ValuesSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(values);
        ValuesSpan {
            start,
            // Checked: a silent wrap here would alias someone else's
            // entries through a corrupted span length.
            len: u16::try_from(values.len()).expect("span exceeds u16 entries"),
        }
    }

    pub fn get(&self, values: ValuesSpan) -> &[ValueId] {
        &self.0[values.start as usize..(values.start as usize + values.len as usize)]
    }

    /// Raw block append for region interface/node blocks, which address
    /// their segments through (start, len) fields on Region rather than
    /// a ValuesSpan. Returns the start of the appended run.
    pub(crate) fn extend(&mut self, values: &[ValueId]) -> u32 {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(values);
        start
    }

    /// A raw (start, len) slice, for region block segments.
    pub(crate) fn slice(&self, start: u32, len: usize) -> &[ValueId] {
        &self.0[start as usize..start as usize + len]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValuesSpan {
    pub start: u32,
    pub len: u16,
}

#[derive(Debug, Clone, Default)]
pub struct RegionPool(Vec<RegionId>);

impl RegionPool {
    pub fn push_slice(&mut self, regions: &[RegionId]) -> RegionsSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(regions);
        RegionsSpan {
            start,
            // Checked like ValuePool: a silent wrap would alias someone
            // else's entries through a corrupted span length.
            len: u16::try_from(regions.len()).expect("span exceeds u16 entries"),
        }
    }

    pub fn get(&self, span: RegionsSpan) -> &[RegionId] {
        &self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }

    /// Mutable view of a span's contents, for the copy-then-remap-in-
    /// place pattern. Every span is uniquely owned by the field holding
    /// it (`push_slice` always appends), so an owner mutating its span
    /// never aliases another owner's.
    pub fn get_mut(&mut self, span: RegionsSpan) -> &mut [RegionId] {
        &mut self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionsSpan {
    pub start: u32,
    pub len: u16,
}

#[derive(Debug, Clone, Default)]
pub struct U32Pool(Vec<u32>);

impl U32Pool {
    pub fn push_slice(&mut self, values: &[u32]) -> U32Span {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(values);
        U32Span {
            start,
            // Checked like ValuePool: see RegionPool::push_slice.
            len: u16::try_from(values.len()).expect("span exceeds u16 entries"),
        }
    }

    pub fn get(&self, values: U32Span) -> &[u32] {
        &self.0[values.start as usize..(values.start as usize + values.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct U32Span {
    pub start: u32,
    pub len: u16,
}

/// One arm of a [`ValueKind::Match`]: an integer input value and the control
/// alternative it selects. Stored in [`MatchArmPool`] so `ValueKind` stays a
/// span (all-`Copy`) rather than carrying a heap allocation per node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchArm {
    /// An integer value the matched input may take (e.g. a `switch` case value).
    pub value: i64,
    /// The 0-based control alternative this input value selects.
    pub alternative: u32,
}

#[derive(Debug, Clone, Default)]
pub struct MatchArmPool(Vec<MatchArm>);

impl MatchArmPool {
    pub fn push_slice(&mut self, arms: &[MatchArm]) -> MatchArmSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(arms);
        MatchArmSpan {
            start,
            // Checked like ValuePool: reachable from a switch with more
            // than u16::MAX cases, which must fail loudly.
            len: u16::try_from(arms.len()).expect("span exceeds u16 entries"),
        }
    }

    pub fn get(&self, span: MatchArmSpan) -> &[MatchArm] {
        &self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchArmSpan {
    pub start: u32,
    pub len: u16,
}

/// State edge -- a newtype over Value for type safety.
/// Prevents accidentally passing a state where data is expected and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State(pub ValueId);

impl State {
    /// Placeholder for a not-yet-set state slot (regions are created
    /// with this exit state and every finaliser must overwrite it, pure
    /// regions included). Forgetting to set it panics at the first
    /// indexed use instead of silently reading a real value, and the
    /// verifier reports any that survive to a checkpoint.
    pub const INVALID: State = State(ValueId::INVALID);
}

#[derive(Debug, Clone)]
pub struct Region {
    /// The gamma/theta value this region belongs to. The graph only
    /// stores the forward direction during emission (the construct value
    /// does not exist until its regions are finished), so like
    /// `exit_state` this is created as [`ValueId::INVALID`] and stamped
    /// by the construct's finaliser. Region 0 is the function body and
    /// stays owner-less; the verifier enforces both directions.
    pub owner: ValueId,
    pub entry_state: State,
    /// Equal to `entry_state` when the region is pure. A field rather
    /// than a results entry so slot machinery (projections, dead slot
    /// elimination) never has to special-case a state slot. Created as
    /// [`State::INVALID`]; every finaliser must set it explicitly, and
    /// the verifier rejects any region still unset.
    pub exit_state: State,
    /// Start of this region's INTERFACE BLOCK in the value pool:
    /// parameters then results, contiguous, written once when the region
    /// is sealed (construct assembly; define_fn's end for the body).
    /// [`Region::UNSEALED`] until then -- an open region's growing lists
    /// live in the graph's construction scratch and every consumer reads
    /// them through the `region_params`/`region_results`/`region_nodes`
    /// accessors, never these fields directly.
    ///
    /// Parameters are appended on demand during emission (the emitter
    /// captures outer values into a region while its body is being
    /// built), so parameter VALUES interleave with body values in the
    /// value arrays; consumers identify a parameter by its position in
    /// the params segment, never by value-id arithmetic.
    pub interface_start: u32,
    pub params_len: u16,
    pub results_len: u16,
    /// This region's nodes in topological (emission) order, as a span in
    /// the value pool, written at seal. Node IDS need not be ascending
    /// (passes append replacement values at high ids); the list order is
    /// the truth for emission and state order.
    pub nodes_start: u32,
    pub nodes_len: u32,
}

impl Region {
    /// Sentinel for a region whose interface block has not been sealed
    /// yet. Accessors panic on a pool read of an unsealed region instead
    /// of slicing from a bogus offset.
    pub const UNSEALED: u32 = u32::MAX;

    /// A freshly created, open region: owner and exit state stamped by
    /// the finaliser, lists living in construction scratch until seal.
    /// Crate-private so `FunctionGraph::create_region` (which registers
    /// the scratch) stays the single way a region comes to exist.
    pub(crate) fn new_open(entry_state: State) -> Self {
        Region {
            owner: ValueId::INVALID,
            entry_state,
            exit_state: State::INVALID,
            interface_start: Region::UNSEALED,
            params_len: 0,
            results_len: 0,
            nodes_start: 0,
            nodes_len: 0,
        }
    }

    pub fn is_sealed(&self) -> bool {
        self.interface_start != Region::UNSEALED
    }

    /// Write this region's sealed storage into `pool` and stamp the
    /// handles: the interface block (params then results, contiguous)
    /// followed by the nodes block. The single definition of the sealed
    /// layout; construction's seal and compaction's rebuild both go
    /// through it, so a layout change cannot diverge between them.
    pub(crate) fn write_blocks(
        &mut self,
        pool: &mut ValuePool,
        params: &[ValueId],
        results: &[ValueId],
        nodes: &[ValueId],
    ) {
        self.interface_start = pool.extend(params);
        // A pool at exactly u32::MAX entries would hand out a start that
        // collides with the UNSEALED sentinel, leaving the region
        // permanently "open".
        debug_assert!(self.interface_start != Region::UNSEALED);
        pool.extend(results);
        self.nodes_start = pool.extend(nodes);
        self.params_len = u16::try_from(params.len()).expect("region parameter count exceeds u16");
        self.results_len = u16::try_from(results.len()).expect("region result count exceeds u16");
        self.nodes_len = u32::try_from(nodes.len()).expect("region node count exceeds u32");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineHint {
    Never,
    Auto,
    Always,
}

/// ELF/Mach-O symbol visibility -- controls linker behavior for shared libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Visibility {
    /// Symbol is visible to other shared objects
    #[default]
    Default,
    /// Symbol is resolved within the defining shared object only
    Hidden,
    /// Like Hidden but the symbol can be overridden by a Default symbol
    Protected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Like Internal but the symbol is also omitted from the symbol table
    Private,
    Internal,
    External,
    /// Merged with other definitions, discarded if unused
    LinkOnce,
    /// Like LinkOnce but preserves the definition for inlining
    LinkOnceODR,
    /// Can be overridden by a stronger definition
    Weak,
    /// Like Weak but preserves the definition for inlining
    WeakODR,
    /// Available for inlining but not emitted if unused
    AvailableExternally,
}
