use crate::rvsdg::{
    ArithFlags, BinaryOp, CastOp, ConstValue, FCmpPred, ICmpPred, IntrinsicOp, State, UnaryOp,
    Value, ValueId, ValueKind,
    constant::ConstId,
    types::{BOOL, I32, I64, TypeRef},
};

use super::{IntrinsicResult, OverflowResult, RegionBuilder};

impl<'a> RegionBuilder<'a> {
    #[inline]
    pub fn add_const(&mut self, ty: TypeRef, const_val: ConstValue) -> ValueId {
        // TODO: add constant dedupe here
        self.add_value(Value {
            ty,
            kind: ValueKind::Const(const_val),
        })
    }

    #[inline]
    pub fn global_ref(&mut self, global: crate::rvsdg::GlobalId, ptr_type: TypeRef) -> ValueId {
        self.add_value(Value {
            ty: ptr_type,
            kind: ValueKind::GlobalRef(global),
        })
    }

    #[inline]
    pub fn const_pool_ref(&mut self, const_id: ConstId, ty: TypeRef) -> ValueId {
        self.add_value(Value {
            ty,
            kind: ValueKind::ConstPoolRef(const_id),
        })
    }

    #[inline]
    pub fn add_const_i32(&mut self, val: i32) -> ValueId {
        self.add_const(I32, ConstValue::Int(val as i64))
    }

    #[inline]
    pub fn add_const_i64(&mut self, val: i64) -> ValueId {
        self.add_const(I64, ConstValue::Int(val))
    }

    #[inline]
    pub fn unary(&mut self, op: UnaryOp, operand: ValueId, ret_type: TypeRef) -> ValueId {
        self.add_value(Value {
            ty: ret_type,
            kind: ValueKind::Unary { op, operand },
        })
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
        self.add_value(Value {
            ty: ret_type,
            kind: ValueKind::Binary {
                op,
                flags,
                left,
                right,
            },
        })
    }

    #[inline]
    pub fn icmp(&mut self, pred: ICmpPred, left: ValueId, right: ValueId) -> ValueId {
        self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::ICmp { pred, left, right },
        })
    }

    #[inline]
    pub fn fcmp(&mut self, pred: FCmpPred, left: ValueId, right: ValueId) -> ValueId {
        self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::FCmp { pred, left, right },
        })
    }

    #[inline]
    pub fn ternary(
        &mut self,
        condition: ValueId,
        true_val: ValueId,
        false_val: ValueId,
        ret_type: TypeRef,
    ) -> ValueId {
        self.add_value(Value {
            ty: ret_type,
            kind: ValueKind::Ternary {
                condition,
                true_val,
                false_val,
            },
        })
    }

    #[inline]
    pub fn cast(&mut self, op: CastOp, value: ValueId, result_type: TypeRef) -> ValueId {
        self.add_value(Value {
            ty: result_type,
            kind: ValueKind::Cast { op, value },
        })
    }

    /// Emit a stateful intrinsic that produces only a new state (no data output).
    /// Use for MemCopy, MemMove, MemSet, LifetimeStart, LifetimeEnd, Unreachable.
    #[inline]
    pub fn intrinsic_void(&mut self, op: IntrinsicOp, state: State, args: &[ValueId]) -> State {
        let args_span = self.graph.value_pool.push_slice(args);
        let val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Intrinsic {
                op,
                state,
                args: args_span,
            },
        });
        State(val)
    }

    /// Emit an intrinsic that produces one data result plus a new state.
    /// Use for IntAbs, FloatFma, FloatMin, FloatMax, FloatCopySign,
    /// saturating arithmetic, and min/max.
    #[inline]
    pub fn intrinsic(
        &mut self,
        op: IntrinsicOp,
        state: State,
        args: &[ValueId],
        ret_type: TypeRef,
    ) -> IntrinsicResult {
        let args_span = self.graph.value_pool.push_slice(args);
        let val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Intrinsic {
                op,
                state,
                args: args_span,
            },
        });
        let result = self.add_value(Value {
            ty: ret_type,
            kind: ValueKind::Project {
                call: val,
                index: 0,
            },
        });
        IntrinsicResult {
            state: State(val),
            value: result,
        }
    }

    /// Emit an overflow-checked arithmetic intrinsic.
    /// Returns (result, overflow_flag) plus a new state.
    #[inline]
    pub fn intrinsic_overflow(
        &mut self,
        op: IntrinsicOp,
        state: State,
        args: &[ValueId],
        ret_type: TypeRef,
    ) -> OverflowResult {
        let args_span = self.graph.value_pool.push_slice(args);
        let val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Intrinsic {
                op,
                state,
                args: args_span,
            },
        });
        let result = self.add_value(Value {
            ty: ret_type,
            kind: ValueKind::Project {
                call: val,
                index: 0,
            },
        });
        let overflow = self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::Project {
                call: val,
                index: 1,
            },
        });
        OverflowResult {
            state: State(val),
            value: result,
            overflow,
        }
    }
}
