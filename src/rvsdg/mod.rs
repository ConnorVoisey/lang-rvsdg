use crate::rvsdg::{
    func::Function,
    types::{TypeArena, TypeRef},
};

pub mod builder;
pub mod func;
pub mod types;

pub struct RVSDGMod {
    pub types: TypeArena,
    pub values: Vec<Value>,
    pub regions: Vec<Region>,
    pub functions: Vec<Function>,
    pub value_pool: ValuePool,
}

impl RVSDGMod {
    pub fn new() -> Self {
        Self {
            types: TypeArena::default(),
            values: vec![],
            regions: vec![],
            functions: vec![],
            value_pool: ValuePool(vec![]),
        }
    }

    #[inline]
    pub fn get(&self, value: ValueId) -> &Value {
        &self.values[value.0 as usize]
    }
}

/// Primary handle into the IR. Indexes into RVSDGMod::values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

/// Not sure this is needed, it was taken from lang
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(u32);

/// The data associated with a Value in the pool.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value {
    pub ty: TypeRef,
    pub kind: ValueKind,
}

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

/// State edge — a newtype over Value for type safety.
/// Prevents accidentally passing a state where data is expected and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State(pub ValueId);

#[derive(Debug, Clone, Copy)]
pub struct LoadResult {
    pub state: State,
    pub value: ValueId,
}

#[derive(Debug, Clone, Copy)]
pub struct StoreResult {
    pub state: State,
}

#[derive(Debug, Clone, Copy)]
pub struct AllocResult {
    pub state: State,
    pub ptr: ValueId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueKind {
    Const(ConstValue),
    GlobalRef {
        name: SymbolId,
    },
    Binary {
        op: BinaryOp,
        flags: ArithFlags,
        left: ValueId,
        right: ValueId,
    },
    ICmp {
        pred: ICmpPred,
        left: ValueId,
        right: ValueId,
    },
    FCmp {
        pred: FCmpPred,
        left: ValueId,
        right: ValueId,
    },
    // Load
    // Store
    // Alloc
    Lambda {
        region: RegionId,
        func_id: FuncId,
    },
    Theta {
        loop_vars: ValuesSpan,
        state: State,
        region: RegionId,
    },
    Gamma {
        condition: ValueId,
        inputs: ValuesSpan,
        state: State,
        true_region: RegionId,
        false_region: RegionId,
    },
    Phi,
    Call {
        state: State,
        fn_id: FuncId,
        args: ValuesSpan,
    },
    Project {
        call: ValueId,
        index: u16,
    },
    RegionParam {
        index: u32,
        ty: TypeRef,
    },
    RegionResult {
        values: ValuesSpan,
        state: State,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstValue {
    /// Covers i1 through i64. The type on the parent Value determines the width.
    Int(i64),
    /// IEEE 754 bits — stored as u32 to support Eq/Hash.
    F32(u32),
    /// IEEE 754 bits — stored as u64 to support Eq/Hash.
    F64(u64),
    NullPtr,
    /// The result of undefined behavior (e.g. signed overflow with no-wrap flags).
    /// Propagates through operations: `poison + 1 = poison`. Triggers UB if it
    /// reaches a side-effecting operation like a store or branch condition.
    /// LLVM's `undef` is lowered to poison on import — we don't distinguish the two.
    Poison,
}

impl ConstValue {
    pub fn f32_from_native(v: f32) -> Self {
        Self::F32(v.to_bits())
    }

    pub fn f64_from_native(v: f64) -> Self {
        Self::F64(v.to_bits())
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    // Integer arithmetic
    Add,
    Sub,
    Mul,
    SignedDiv,
    UnsignedDiv,
    SignedRem,
    UnsignedRem,
    // Bitwise
    ShiftLeft,
    LogicalShiftRight,
    ArithShiftRight,
    And,
    Or,
    Xor,
    // Floating-point arithmetic
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,
    FloatRem,
}

/// Arithmetic flags. Not all flags apply to all ops —
/// no_signed_wrap/no_unsigned_wrap apply to add/sub/mul/shl,
/// exact applies to udiv/sdiv/lshr/ashr.
/// Unused flags for a given op are simply false.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ArithFlags {
    pub no_signed_wrap: bool,
    pub no_unsigned_wrap: bool,
    /// Exact (no rounding for div/shift)
    pub exact: bool,
}

/// Integer comparison predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ICmpPred {
    Eq,
    Ne,
    UnsignedGt,
    UnsignedGe,
    UnsignedLt,
    UnsignedLe,
    SignedGt,
    SignedGe,
    SignedLt,
    SignedLe,
}

/// Floating-point comparison predicates.
/// "Ordered" means neither operand is NaN. "Unordered" means the comparison
/// returns true if either operand is NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FCmpPred {
    /// Always false
    False,
    OrderedEq,
    OrderedGt,
    OrderedGe,
    OrderedLt,
    OrderedLe,
    OrderedNe,
    /// Both operands are non-NaN
    Ordered,
    UnorderedEq,
    UnorderedGt,
    UnorderedGe,
    UnorderedLt,
    UnorderedLe,
    UnorderedNe,
    /// Either operand is NaN
    Unordered,
    /// Always true
    True,
}

pub struct Region {
    pub id: RegionId,
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
