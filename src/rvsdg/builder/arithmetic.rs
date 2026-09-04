use crate::rvsdg::{
    AliasClassId, ArithFlags, BinaryOp, CastOp, ConstValue, FCmpPred, FuncId, ICmpPred,
    IntrinsicOp, StateKind, UnaryOp, Value, ValueId, ValueKind,
    constant::ConstId,
    memory_alias::origin::MemoryOrigin,
    types::{BOOL, I32, I64, TypeRef},
};

use super::{OverflowResult, RegionBuilder};

impl<'a> RegionBuilder<'a> {
    // Constants and symbol references are region-free: interned
    // module-wide on the graph, one node per distinct value, owned by
    // no region. These builder methods are thin forwards kept so
    // region-building code has everything in one place.

    #[inline]
    pub fn constant(&mut self, ty: TypeRef, const_val: ConstValue) -> ValueId {
        self.graph.intern_const(ty, const_val)
    }

    #[inline]
    pub fn global_ref(&mut self, global: crate::rvsdg::GlobalId, ptr_type: TypeRef) -> ValueId {
        self.graph.intern_global_ref(global, ptr_type)
    }

    #[inline]
    pub fn func_addr(&mut self, func_id: FuncId, ptr_type: TypeRef) -> ValueId {
        self.graph.intern_func_addr(func_id, ptr_type)
    }

    #[inline]
    pub fn const_pool_ref(&mut self, const_id: ConstId, ty: TypeRef) -> ValueId {
        // The graph cannot see the constant pool, so the base global of
        // a pointer-typed pool constant resolves here.
        let origin = if matches!(ty, TypeRef::Ptr(_)) {
            MemoryOrigin::of_constant(&self.module_tables.constants, const_id)
        } else {
            MemoryOrigin::None
        };
        self.graph.intern_const_pool_ref(const_id, ty, origin)
    }

    #[inline]
    pub fn const_i32(&mut self, val: i32) -> ValueId {
        self.constant(I32, ConstValue::Int(val as i64))
    }

    #[inline]
    pub fn const_i64(&mut self, val: i64) -> ValueId {
        self.constant(I64, ConstValue::Int(val))
    }

    #[inline]
    pub fn unary(&mut self, op: UnaryOp, operand: ValueId, ret_type: TypeRef) -> ValueId {
        self.add_value(
            Value {
                ty: ret_type,
                kind: ValueKind::Unary { op, operand },
            },
            MemoryOrigin::None,
        )
    }

    #[inline]
    pub fn binary(
        &mut self,
        op: BinaryOp,
        flags: ArithFlags,
        left: ValueId,
        right: ValueId,
        ret_type: TypeRef,
    ) -> ValueId {
        // Pointer arithmetic never comes through here (that is
        // ptr_offset); integer results carry no origin.
        self.add_value(
            Value {
                ty: ret_type,
                kind: ValueKind::Binary {
                    op,
                    flags,
                    left,
                    right,
                },
            },
            MemoryOrigin::None,
        )
    }

    #[inline]
    pub fn icmp(&mut self, pred: ICmpPred, left: ValueId, right: ValueId) -> ValueId {
        self.add_value(
            Value {
                ty: BOOL,
                kind: ValueKind::ICmp { pred, left, right },
            },
            MemoryOrigin::None,
        )
    }

    #[inline]
    pub fn fcmp(&mut self, pred: FCmpPred, left: ValueId, right: ValueId) -> ValueId {
        self.add_value(
            Value {
                ty: BOOL,
                kind: ValueKind::FCmp { pred, left, right },
            },
            MemoryOrigin::None,
        )
    }

    #[inline]
    pub fn ternary(
        &mut self,
        condition: ValueId,
        true_val: ValueId,
        false_val: ValueId,
        ret_type: TypeRef,
    ) -> ValueId {
        // A pointer select is a JOIN: link one side now (never a copy,
        // the one rule); the other side becomes a join event once the
        // event scratch lands, and resolution widens disagreements.
        let is_ptr = matches!(ret_type, TypeRef::Ptr(_));
        let origin = if is_ptr {
            MemoryOrigin::Derived(true_val)
        } else {
            MemoryOrigin::None
        };
        let result = self.add_value(
            Value {
                ty: ret_type,
                kind: ValueKind::Ternary {
                    condition,
                    true_val,
                    false_val,
                },
            },
            origin,
        );
        if is_ptr {
            self.graph.record_join_event(result, false_val);
        }
        result
    }

    #[inline]
    pub fn cast(&mut self, op: CastOp, value: ValueId, result_type: TypeRef) -> ValueId {
        let origin = match op {
            // The same pointer under a different name.
            CastOp::Bitcast if matches!(result_type, TypeRef::Ptr(_)) => {
                MemoryOrigin::Derived(value)
            }
            // A laundered pointer: origins cannot follow integers.
            CastOp::IntToPtr => MemoryOrigin::Unknown,
            // Everything else produces non-pointers.
            _ => MemoryOrigin::None,
        };
        if matches!(op, CastOp::PtrToInt) {
            // The address becomes a number origins cannot follow.
            self.graph.record_escape_event(value);
        }
        self.add_value(
            Value {
                ty: result_type,
                kind: ValueKind::Cast { op, value },
            },
            origin,
        )
    }

    // Stateful intrinsics thread as writes: the input state (pending
    // reads flattened) comes from the region's scratch registers.

    /// Emit a stateful intrinsic that produces no data output.
    /// Use for MemCopy, MemMove, MemSet, LifetimeStart, LifetimeEnd, Unreachable.
    #[inline]
    pub fn intrinsic_void(&mut self, op: IntrinsicOp, args: &[ValueId]) {
        let args_span = self.graph.value_pool.push_slice(args);
        let out_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::Intrinsic {
                        op,
                        state,
                        args: args_span,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(out_state.0);
    }

    /// Emit an intrinsic that produces one data result, returned.
    /// Use for IntAbs, FloatFma, FloatMin, FloatMax, FloatCopySign,
    /// saturating arithmetic, and min/max.
    #[inline]
    pub fn intrinsic(&mut self, op: IntrinsicOp, args: &[ValueId], ret_type: TypeRef) -> ValueId {
        let args_span = self.graph.value_pool.push_slice(args);
        let out_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::Intrinsic {
                        op,
                        state,
                        args: args_span,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(out_state.0);
        self.add_value(
            Value {
                ty: ret_type,
                kind: ValueKind::Project {
                    call: out_state.0,
                    index: 0,
                },
            },
            MemoryOrigin::unknown_if_ptr(ret_type),
        )
    }

    /// Emit an overflow-checked arithmetic intrinsic.
    /// Returns the result value and the overflow flag.
    #[inline]
    pub fn intrinsic_overflow(
        &mut self,
        op: IntrinsicOp,
        args: &[ValueId],
        ret_type: TypeRef,
    ) -> OverflowResult {
        let args_span = self.graph.value_pool.push_slice(args);
        let out_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::Intrinsic {
                        op,
                        state,
                        args: args_span,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(out_state.0);
        let result = self.add_value(
            Value {
                ty: ret_type,
                kind: ValueKind::Project {
                    call: out_state.0,
                    index: 0,
                },
            },
            MemoryOrigin::None,
        );
        let overflow = self.add_value(
            Value {
                ty: BOOL,
                kind: ValueKind::Project {
                    call: out_state.0,
                    index: 1,
                },
            },
            MemoryOrigin::None,
        );
        OverflowResult {
            value: result,
            overflow,
        }
    }
}
