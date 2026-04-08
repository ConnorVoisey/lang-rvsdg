use crate::rvsdg::{
    FuncId, GlobalId, RegionId, RegionsSpan, State, ValueId, ValuesSpan,
    constant::ConstId,
    ops::{
        ArithFlags, AtomicRMWOp, BinaryOp, CastOp, FCmpPred, ICmpPred, IntrinsicOp, MemoryOrdering,
        UnaryOp,
    },
    types::TypeRef,
};

/// The data associated with a Value in the pool.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value {
    pub ty: TypeRef,
    pub kind: ValueKind,
}

// Size: 32 bytes. Driven by Load/Store/AtomicLoad variants at 25 bytes payload.
// Most variants are 4-16 bytes but boxing the large ones would add pointer chases
// on the most frequently accessed operations — not worth the tradeoff.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueKind {
    Const(ConstValue),
    /// Reference to a constant in the constant pool (aggregates, strings, etc.)
    ConstPoolRef(ConstId),
    /// Produces a pointer to a global variable.
    GlobalRef(GlobalId),
    /// Produces a function pointer from a known function.
    FuncAddr(FuncId),
    Unary {
        op: UnaryOp,
        operand: ValueId,
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
    /// Branch-free conditional value selection (LLVM's `select`).
    /// `condition ? true_val : false_val` — no control flow, no state edge.
    Ternary {
        condition: ValueId,
        true_val: ValueId,
        false_val: ValueId,
    },
    Cast {
        op: CastOp,
        value: ValueId,
    },
    /// Read a single lane from a vector by index.
    ExtractLane {
        vector: ValueId,
        index: ValueId,
    },
    /// Write a single lane into a vector, producing a new vector.
    InsertLane {
        vector: ValueId,
        index: ValueId,
        value: ValueId,
    },
    /// Rearrange lanes from two vectors according to a mask.
    /// Mask entries are constant indices: 0..N select from `left`, N..2N from `right`.
    ShuffleLanes {
        left: ValueId,
        right: ValueId,
        mask: ValuesSpan,
    },
    /// Read a field from a by-value aggregate (struct or array).
    /// Indices are compile-time constants that walk nested aggregates.
    ExtractField {
        aggregate: ValueId,
        /// Constant index path (e.g. [0, 1] for the second field of the first nested struct)
        indices: ValuesSpan,
    },
    /// Write a field into a by-value aggregate, producing a new aggregate.
    /// Indices are compile-time constants that walk nested aggregates.
    InsertField {
        aggregate: ValueId,
        value: ValueId,
        /// Constant index path
        indices: ValuesSpan,
    },
    /// Compute a pointer to a field or element within an aggregate.
    /// LLVM's `getelementptr` — indices walk through nested structs/arrays.
    PtrOffset {
        base: ValueId,
        /// The type being indexed into (the pointee type of base)
        base_type: TypeRef,
        /// Index values — struct field indices are constants, array indices are dynamic
        indices: ValuesSpan,
        /// UB if the result is out of bounds (enables pointer arithmetic optimizations)
        inbounds: bool,
    },
    /// Read a value from memory. The value node itself is the output state;
    /// use Project { index: 0 } to get the loaded value.
    Load {
        state: State,
        addr: ValueId,
        /// The type being loaded
        loaded_type: TypeRef,
        /// Alignment in bytes (None = natural alignment for the type)
        align: Option<u32>,
        /// Volatile loads cannot be reordered, eliminated, or duplicated
        volatile: bool,
    },
    /// Write a value to memory. The value node itself is the output state.
    Store {
        state: State,
        addr: ValueId,
        value: ValueId,
        /// Alignment in bytes (None = natural alignment for the type)
        align: Option<u32>,
        /// Volatile stores cannot be reordered, eliminated, or duplicated
        volatile: bool,
    },
    /// Stack allocation. The value node itself is the output state;
    /// use Project { index: 0 } to get the pointer.
    Alloca {
        state: State,
        /// Type of each element
        elem_type: TypeRef,
        /// Number of elements (usually a constant 1)
        count: ValueId,
    },
    /// Atomic load. Output state is the node; Project { index: 0 } for the value.
    AtomicLoad {
        state: State,
        addr: ValueId,
        loaded_type: TypeRef,
        ordering: MemoryOrdering,
        align: Option<u32>,
    },
    /// Atomic store. The node itself is the output state.
    AtomicStore {
        state: State,
        addr: ValueId,
        value: ValueId,
        ordering: MemoryOrdering,
        align: Option<u32>,
    },
    /// Atomic read-modify-write. Output state is the node;
    /// Project { index: 0 } for the old value.
    AtomicReadModifyWrite {
        state: State,
        addr: ValueId,
        value: ValueId,
        op: AtomicRMWOp,
        ordering: MemoryOrdering,
    },
    /// Atomic compare-and-swap. Output state is the node;
    /// Project { index: 0 } for the old value, Project { index: 1 } for success flag.
    CompareAndSwap {
        state: State,
        addr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    },
    /// Memory fence. The node itself is the output state.
    Fence {
        state: State,
        ordering: MemoryOrdering,
    },
    /// Convert poison/undef to an arbitrary but fixed value.
    /// Pure — no state edge needed.
    Freeze {
        value: ValueId,
    },
    /// Built-in memory/arithmetic intrinsics that don't branch.
    Intrinsic {
        op: IntrinsicOp,
        state: State,
        args: ValuesSpan,
    },
    Lambda {
        region: RegionId,
        func_id: FuncId,
    },
    Theta {
        loop_vars: ValuesSpan,
        condition: ValueId,
        state: State,
        region_id: RegionId,
    },
    /// N-way conditional branch. The condition selects which region to execute:
    /// 0 → first region, 1 → second, etc. For a 2-way if/else, condition is a bool.
    Gamma {
        condition: ValueId,
        inputs: ValuesSpan,
        state: State,
        /// One region per branch, all must produce the same number/types of results
        regions: RegionsSpan,
    },
    Phi {
        /// The region containing the mutually recursive definitions
        region: RegionId,
        /// The recursion variables (function references available inside the phi body)
        rv_count: u16,
    },
    Call {
        state: State,
        fn_id: FuncId,
        args: ValuesSpan,
    },
    /// Indirect call through a function pointer.
    CallIndirect {
        state: State,
        callee: ValueId,
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
