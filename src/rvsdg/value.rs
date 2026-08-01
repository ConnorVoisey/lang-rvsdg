use crate::rvsdg::{
    FuncId, GlobalId, MatchArmSpan, RegionId, RegionsSpan, State, U32Span, ValueId, ValuesSpan,
    constant::ConstId,
    func::SignatureId,
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

// A Value (this kind plus its 8-byte TypeRef) measures 40 bytes; the
// census's memory-budget table tracks the real figure. The size is
// driven by the memory-op variants; most variants are 4-16 bytes, but
// boxing the large ones would add pointer chases on the most frequently
// accessed operations -- not worth the tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// `condition ? true_val : false_val` -- no control flow, no state edge.
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
        indices: U32Span,
    },
    /// Write a field into a by-value aggregate, producing a new aggregate.
    /// Indices are compile-time constants that walk nested aggregates.
    InsertField {
        aggregate: ValueId,
        value: ValueId,
        /// Constant index path
        indices: U32Span,
    },
    /// Compute a pointer to a field or element within an aggregate.
    /// LLVM's `getelementptr` -- indices walk through nested structs/arrays.
    PtrOffset {
        base: ValueId,
        /// The type being indexed into (the pointee type of base)
        base_type: TypeRef,
        /// Index values -- struct field indices are constants, array indices are dynamic
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
        /// Alignment in bytes (None = natural alignment for the type).
        /// MUST be carried: initialising stores/memcpys keep their own
        /// alignment claims, and a slot laid out below what they claim
        /// faults under the backend's aligned-SSE expansions.
        align: Option<u32>,
    },
    /// Atomic load. Output state is the node; Project { index: 0 } for the value.
    AtomicLoad {
        state: State,
        addr: ValueId,
        loaded_type: TypeRef,
        ordering: MemoryOrdering,
        align: Option<u32>,
        volatile: bool,
    },
    /// Atomic store. The node itself is the output state.
    AtomicStore {
        state: State,
        addr: ValueId,
        value: ValueId,
        ordering: MemoryOrdering,
        align: Option<u32>,
        volatile: bool,
    },
    /// Atomic read-modify-write. Output state is the node;
    /// Project { index: 0 } for the old value.
    AtomicReadModifyWrite {
        state: State,
        addr: ValueId,
        value: ValueId,
        op: AtomicRMWOp,
        ordering: MemoryOrdering,
        volatile: bool,
    },
    /// Atomic compare-and-swap. Output state is the node;
    /// Project { index: 0 } for the old value, Project { index: 1 } for success flag.
    /// Always strong: a strong compare-and-swap never fails spuriously,
    /// which is a valid implementation of LLVM's `weak` form, so the weak
    /// flag is dropped at parse time.
    CompareAndSwap {
        state: State,
        addr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
        volatile: bool,
    },
    /// Memory fence. The node itself is the output state.
    Fence {
        state: State,
        ordering: MemoryOrdering,
    },
    /// Convert poison/undef to an arbitrary but fixed value.
    /// Pure -- no state edge needed.
    Freeze {
        value: ValueId,
    },
    /// Match an integer `input` into a control/predicate value (Bahmann et al.
    /// 2015 section 2.2): the "match" that turns an integer condition
    /// into a predicate enumerating alternatives. The produced value has type
    /// `Control(alternatives)`. Each arm in `arms` maps a specific input value
    /// to a control alternative (0-based); any input value not listed maps to
    /// `default`. gamma/theta consume the resulting control value -- never the raw
    /// integer -- which keeps predicates in the single-use form perfect
    /// reconstruction requires (Def 2.6) and records the original case values
    /// so the source branch/switch is recoverable.
    Match {
        input: ValueId,
        arms: MatchArmSpan,
        default: u32,
        /// Number of control alternatives (matches the `Control(n)` type).
        alternatives: u32,
    },
    /// Built-in memory/arithmetic intrinsics that don't branch.
    Intrinsic {
        op: IntrinsicOp,
        state: State,
        args: ValuesSpan,
    },
    Theta {
        loop_vars: ValuesSpan,
        condition: ValueId,
        state: State,
        region_id: RegionId,
    },
    /// N-way conditional branch. The condition selects which region to execute:
    /// 0 -> first region, 1 -> second, etc. For a 2-way if/else, condition is a bool.
    Gamma {
        condition: ValueId,
        inputs: ValuesSpan,
        state: State,
        /// One region per branch, all must produce the same number/types of results
        regions: RegionsSpan,
    },
    /// Direct call to a known function. The call site carries its own
    /// interned ABI [`Signature`](crate::rvsdg::func::Signature), same as
    /// an indirect call: LLVM attributes live on call sites as well as
    /// declarations, and for a variadic call the site is the ONLY place
    /// the variadic actual arguments' ABI attributes exist (e.g. byval on
    /// a struct passed through `...`) -- the declaration has no parameter
    /// entries for them.
    Call {
        state: State,
        fn_id: FuncId,
        sig: SignatureId,
        args: ValuesSpan,
    },
    /// Indirect call through a function pointer. The callee's full ABI
    /// signature (function type, parameter/return attributes, calling
    /// convention) is stored here rather than derived from the callee
    /// value: pointers are opaque (no pointee type), so the call site is
    /// the only place the signature exists -- the same reason LLVM call
    /// instructions carry their own function type.
    CallIndirect {
        state: State,
        callee: ValueId,
        sig: SignatureId,
        args: ValuesSpan,
    },
    Project {
        call: ValueId,
        index: u16,
    },
    RegionParam {
        index: u32,
        ty: TypeRef,
        /// The region this value is a parameter of. Regions do not own
        /// their parameter values in the global array, so the back link
        /// lives here; stamped at creation, verified against the
        /// region's params list.
        region: RegionId,
    },
    RegionResult {
        values: ValuesSpan,
        state: State,
    },
}

impl ValueKind {
    /// Region-free values denote the same thing in every region:
    /// constants and symbol references. They are interned module-wide
    /// (one node per distinct value), belong to NO region's node list,
    /// and the scope rules exempt them from the values-flow-through-
    /// edges requirement -- the emitter materialises LLVM constants for
    /// them on demand, which needs no dominance.
    pub fn is_region_free(&self) -> bool {
        matches!(
            self,
            ValueKind::Const(_)
                | ValueKind::ConstPoolRef(_)
                | ValueKind::GlobalRef(_)
                | ValueKind::FuncAddr(_)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstValue {
    /// Covers i1 through i64. The type on the parent Value determines the width.
    Int(i64),
    /// IEEE 754 bits -- stored as u32 to support Eq/Hash.
    F32(u32),
    /// IEEE 754 bits -- stored as u64 to support Eq/Hash.
    F64(u64),
    NullPtr,
    /// The result of undefined behavior (e.g. signed overflow with no-wrap flags).
    /// Propagates through operations: `poison + 1 = poison`. Triggers UB if it
    /// reaches a side-effecting operation like a store or branch condition.
    /// LLVM's `undef` is lowered to poison on import -- we don't distinguish the two.
    Poison,
}

impl ConstValue {
    pub fn f32_from_native(v: f32) -> Self {
        Self::F32(v.to_bits())
    }

    pub fn f64_from_native(v: f64) -> Self {
        Self::F64(v.to_bits())
    }
}
