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
pub mod region;
pub mod state;
pub mod types;
pub mod value;
pub mod verify;

pub use constant::{ConstId, ConstIdPool, ConstIdsSpan, ConstantDef, ConstantKind, ConstantPool};
pub use global::{GlobalDef, GlobalInit, ThreadLocalMode};
pub use ops::{
    ArithFlags, AtomicRMWOp, BinaryOp, CastOp, FCmpPred, ICmpPred, IntrinsicOp, MemoryOrdering,
    UnaryOp,
};
pub use state::{AliasClassId, State, StateKind};
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

    pub fn push_iter(&mut self, values: impl Iterator<Item = ValueId>) -> ValuesSpan {
        let start = self.0.len() as u32;
        let mut len = 0;
        for val in values {
            self.0.push(val);
            len += 1;
        }
        ValuesSpan {
            start,
            // Checked: a silent wrap here would alias someone else's
            // entries through a corrupted span length.
            len: u16::try_from(len).expect("span exceeds u16 entries"),
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

    /// Mutable view of a block segment. Every block is uniquely owned by
    /// the Region holding its handles, so an owner mutating its segment
    /// never aliases another owner's.
    pub(crate) fn slice_mut(&mut self, start: u32, len: usize) -> &mut [ValueId] {
        &mut self.0[start as usize..start as usize + len]
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
    pub(crate) fn get_mut(&mut self, span: RegionsSpan) -> &mut [RegionId] {
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
