#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Negate a float (-x)
    FloatNeg,
    /// Bitwise NOT (~x)
    BitNot,
    /// Count leading zeros
    CountLeadingZeros,
    /// Count trailing zeros
    CountTrailingZeros,
    /// Count set bits (popcount)
    CountOnes,
    /// Byte-swap (endian reversal)
    ByteSwap,
    /// Bit reversal
    BitReverse,
    /// Floating-point absolute value
    FloatAbs,
    /// Floating-point floor
    FloatFloor,
    /// Floating-point ceil
    FloatCeil,
    /// Floating-point round to nearest
    FloatRound,
    /// Floating-point square root
    FloatSqrt,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastOp {
    /// Integer sign-extend to wider type (i8 -> i32)
    SignExtend,
    /// Integer zero-extend to wider type (i8 -> i32, high bits = 0)
    ZeroExtend,
    /// Integer truncate to narrower type (i64 -> i32)
    Truncate,
    /// Float extend to wider type (f32 -> f64)
    FloatExtend,
    /// Float truncate to narrower type (f64 -> f32)
    FloatTruncate,
    /// Signed integer to float (i32 -> f64)
    SignedToFloat,
    /// Unsigned integer to float (u32 -> f64)
    UnsignedToFloat,
    /// Float to signed integer (f64 -> i32)
    FloatToSigned,
    /// Float to unsigned integer (f64 -> u32)
    FloatToUnsigned,
    /// Pointer to integer
    PtrToInt,
    /// Integer to pointer
    IntToPtr,
    /// Reinterpret bits without changing them (e.g. ptr addrspace cast)
    Bitcast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryOrdering {
    /// No ordering constraint (non-atomic context or relaxed)
    Relaxed,
    /// Reads after this fence see all writes before a Release in another thread
    Acquire,
    /// Writes before this fence are visible to Acquire reads in another thread
    Release,
    /// Both Acquire and Release
    AcquireRelease,
    /// Total ordering across all threads
    SequentiallyConsistent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicRMWOp {
    /// Replace with new value
    Exchange,
    /// Integer add
    Add,
    /// Integer sub
    Sub,
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NAND
    Nand,
    /// Signed integer max
    SignedMax,
    /// Signed integer min
    SignedMin,
    /// Unsigned integer max
    UnsignedMax,
    /// Unsigned integer min
    UnsignedMin,
    /// Float add
    FloatAdd,
    /// Float sub
    FloatSub,
    /// Float max
    FloatMax,
    /// Float min
    FloatMin,
}

/// Built-in operations that lower to known instruction sequences rather than calls.
/// All memory intrinsics take state and produce state. Args layout is documented per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicOp {
    // ── Memory operations ───────────────────────────────────────
    /// Copy non-overlapping memory. Args: [dst, src, len]
    MemCopy,
    /// Copy possibly-overlapping memory. Args: [dst, src, len]
    MemMove,
    /// Fill memory with a byte value. Args: [dst, val, len]
    MemSet,

    // ── Overflow-checked arithmetic (result + overflow flag) ────
    /// Signed add with overflow. Args: [lhs, rhs] → (result, overflow_bit)
    SignedAddOverflow,
    /// Unsigned add with overflow. Args: [lhs, rhs] → (result, overflow_bit)
    UnsignedAddOverflow,
    /// Signed sub with overflow. Args: [lhs, rhs] → (result, overflow_bit)
    SignedSubOverflow,
    /// Unsigned sub with overflow. Args: [lhs, rhs] → (result, overflow_bit)
    UnsignedSubOverflow,
    /// Signed mul with overflow. Args: [lhs, rhs] → (result, overflow_bit)
    SignedMulOverflow,
    /// Unsigned mul with overflow. Args: [lhs, rhs] → (result, overflow_bit)
    UnsignedMulOverflow,

    // ── Saturating arithmetic ──────────────────────────────────
    /// Signed add clamped to min/max. Args: [lhs, rhs]
    SignedAddSaturate,
    /// Unsigned add clamped to max. Args: [lhs, rhs]
    UnsignedAddSaturate,
    /// Signed sub clamped to min/max. Args: [lhs, rhs]
    SignedSubSaturate,
    /// Unsigned sub clamped to 0. Args: [lhs, rhs]
    UnsignedSubSaturate,

    // ── Min / Max ──────────────────────────────────────────────
    /// Signed integer minimum. Args: [lhs, rhs]
    SignedMin,
    /// Signed integer maximum. Args: [lhs, rhs]
    SignedMax,
    /// Unsigned integer minimum. Args: [lhs, rhs]
    UnsignedMin,
    /// Unsigned integer maximum. Args: [lhs, rhs]
    UnsignedMax,

    // ── Misc ───────────────────────────────────────────────────
    /// Absolute value of a signed integer. Args: [value]
    IntAbs,
    /// Fused multiply-add (a*b + c). Args: [a, b, c]
    FloatFma,
    /// Select the arithmetic minimum of two floats. Args: [lhs, rhs]
    FloatMin,
    /// Select the arithmetic maximum of two floats. Args: [lhs, rhs]
    FloatMax,
    /// Copy sign of second float onto first. Args: [mag, sign]
    FloatCopySign,
    /// Lifetime start marker for alloca. Args: [ptr]
    LifetimeStart,
    /// Lifetime end marker for alloca. Args: [ptr]
    LifetimeEnd,
    /// Compiler hint that a condition is expected true/false. Args: [condition, expected]
    Expect,
    /// Marks a point as unreachable. Args: []
    Unreachable,
}
