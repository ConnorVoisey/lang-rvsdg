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
use smallvec::SmallVec;
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

/// Primary handle into the IR. Indexes into RVSDGMod::values.
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
            len: values.len() as u16,
        }
    }

    pub fn get(&self, values: ValuesSpan) -> &[ValueId] {
        &self.0[values.start as usize..(values.start as usize + values.len as usize)]
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
            len: regions.len() as u16,
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
            len: values.len() as u16,
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
            len: arms.len() as u16,
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
    /// The region's parameters, in input order. An explicit list (not a
    /// contiguous span) because construction appends parameters on demand:
    /// the emitter captures outer values into a region while its body is
    /// being built, so parameter values interleave with body values in the
    /// global value array. Consumers identify a parameter by its position
    /// here, never by value-id arithmetic.
    pub params: SmallVec<[ValueId; 8]>,
    /// The lambda/gamma/theta/phi value this region belongs to. The
    /// graph only stores the forward direction during emission (the
    /// construct value does not exist until its regions are finished),
    /// so like `exit_state` this is created as [`ValueId::INVALID`] and
    /// stamped by the construct's finaliser; the verifier rejects any
    /// region left unset.
    pub owner: ValueId,
    pub entry_state: State,
    /// Equal to `entry_state` when the region is pure. A field rather
    /// than a results-span entry so slot machinery (projections, phis,
    /// dead slot elimination) never has to special-case a state slot.
    /// Created as [`State::INVALID`]; every finaliser must set it
    /// explicitly, and the verifier rejects any region still unset.
    pub exit_state: State,
    pub results: ValuesSpan,
    /// All values in this region (in topo order)
    pub nodes: Vec<ValueId>,
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
