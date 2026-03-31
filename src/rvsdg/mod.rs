pub mod builder;
pub mod constant;
pub mod func;
pub mod global;
pub mod ops;
pub mod types;
pub mod value;

pub use constant::{ConstId, ConstIdPool, ConstIdsSpan, ConstantDef, ConstantKind, ConstantPool};
pub use global::{GlobalDef, GlobalInit, GlobalLinkage};
pub use ops::{
    ArithFlags, AtomicRMWOp, BinaryOp, CastOp, FCmpPred, ICmpPred, IntrinsicOp, MemoryOrdering,
    UnaryOp,
};
pub use value::{ConstValue, Value, ValueKind};

use func::Function;
use types::{TypeArena, TypeRef};

pub use target_lexicon::Triple;

#[derive(Debug)]
pub struct RVSDGMod {
    /// Target triple (e.g. x86_64-unknown-linux-gnu)
    pub target: Triple,
    /// LLVM data layout string — encodes pointer sizes, alignments, endianness
    /// for the target. Preserved verbatim for roundtripping through LLVM.
    pub data_layout: String,
    pub types: TypeArena,
    pub values: Vec<Value>,
    pub regions: Vec<Region>,
    pub functions: Vec<Function>,
    pub globals: Vec<GlobalDef>,
    pub constants: ConstantPool,
    pub value_pool: ValuePool,
    pub region_pool: RegionPool,
}

impl RVSDGMod {
    pub fn new(target: Triple, data_layout: String) -> Self {
        Self {
            target,
            data_layout,
            types: TypeArena::default(),
            values: vec![],
            regions: vec![],
            functions: vec![],
            globals: vec![],
            constants: ConstantPool::default(),
            value_pool: ValuePool(vec![]),
            region_pool: RegionPool(vec![]),
        }
    }

    /// Create a module targeting the host platform with an empty data layout.
    pub fn new_host() -> Self {
        Self::new(Triple::host(), String::new())
    }

    #[inline]
    pub fn get(&self, value: ValueId) -> &Value {
        &self.values[value.0 as usize]
    }

    #[inline]
    pub fn get_region(&self, region_id: RegionId) -> &Region {
        &self.regions[region_id.0 as usize]
    }

    pub fn define_global(
        &mut self,
        name: String,
        ty: TypeRef,
        initializer: GlobalInit,
        is_constant: bool,
        linkage: GlobalLinkage,
    ) -> GlobalId {
        let id = GlobalId(self.globals.len() as u32);
        self.globals.push(GlobalDef {
            name,
            ty,
            initializer,
            is_constant,
            linkage,
            alignment: None,
            section: None,
            visibility: Visibility::default(),
        });
        id
    }

    #[inline]
    pub fn get_global(&self, id: GlobalId) -> &GlobalDef {
        &self.globals[id.0 as usize]
    }

    #[inline]
    pub fn get_region_mut(&mut self, region_id: RegionId) -> &mut Region {
        &mut self.regions[region_id.0 as usize]
    }
}

/// Primary handle into the IR. Indexes into RVSDGMod::values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValuesSpan {
    pub start: u32,
    pub len: u16,
}

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionsSpan {
    pub start: u32,
    pub len: u16,
}

/// State edge — a newtype over Value for type safety.
/// Prevents accidentally passing a state where data is expected and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State(pub ValueId);

#[derive(Debug, Clone)]
pub struct Region {
    pub params: ValuesSpan,
    pub entry_state: State,
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

/// ELF/Mach-O symbol visibility — controls linker behavior for shared libraries.
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
