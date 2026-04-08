use crate::rvsdg::{
    ArithFlags, BinaryOp, RVSDGMod, ValueId,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
};
use inkwell::{builder::BuilderError, values::BasicValueEnum};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_binary<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        op: BinaryOp,
        flags: ArithFlags,
        left: ValueId,
        right: ValueId,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let lhs = self.expect_value(llvm_builder, mapper, rvsdg_func, left)?;
        let rhs = self.expect_value(llvm_builder, mapper, rvsdg_func, right)?;
        let lhs_int = || lhs.into_int_value();
        let rhs_int = || rhs.into_int_value();
        let lhs_float = || lhs.into_float_value();
        let rhs_float = || rhs.into_float_value();
        let b = &llvm_builder.builder;

        Ok(match op {
            BinaryOp::Add => BasicValueEnum::IntValue(if flags.no_signed_wrap {
                b.build_int_nsw_add(lhs_int(), rhs_int(), "add")?
            } else if flags.no_unsigned_wrap {
                b.build_int_nuw_add(lhs_int(), rhs_int(), "add")?
            } else {
                b.build_int_add(lhs_int(), rhs_int(), "add")?
            }),
            BinaryOp::Sub => BasicValueEnum::IntValue(if flags.no_signed_wrap {
                b.build_int_nsw_sub(lhs_int(), rhs_int(), "sub")?
            } else if flags.no_unsigned_wrap {
                b.build_int_nuw_sub(lhs_int(), rhs_int(), "sub")?
            } else {
                b.build_int_sub(lhs_int(), rhs_int(), "sub")?
            }),
            BinaryOp::Mul => BasicValueEnum::IntValue(if flags.no_signed_wrap {
                b.build_int_nsw_mul(lhs_int(), rhs_int(), "mul")?
            } else if flags.no_unsigned_wrap {
                b.build_int_nuw_mul(lhs_int(), rhs_int(), "mul")?
            } else {
                b.build_int_mul(lhs_int(), rhs_int(), "mul")?
            }),
            BinaryOp::SignedDiv => BasicValueEnum::IntValue(if flags.exact {
                b.build_int_exact_signed_div(lhs_int(), rhs_int(), "sdiv")?
            } else {
                b.build_int_signed_div(lhs_int(), rhs_int(), "sdiv")?
            }),
            BinaryOp::UnsignedDiv => {
                BasicValueEnum::IntValue(b.build_int_unsigned_div(lhs_int(), rhs_int(), "udiv")?)
            }
            BinaryOp::SignedRem => {
                BasicValueEnum::IntValue(b.build_int_signed_rem(lhs_int(), rhs_int(), "srem")?)
            }
            BinaryOp::UnsignedRem => {
                BasicValueEnum::IntValue(b.build_int_unsigned_rem(lhs_int(), rhs_int(), "urem")?)
            }
            BinaryOp::ShiftLeft => {
                BasicValueEnum::IntValue(b.build_left_shift(lhs_int(), rhs_int(), "shl")?)
            }
            BinaryOp::LogicalShiftRight => BasicValueEnum::IntValue(b.build_right_shift(
                lhs_int(),
                rhs_int(),
                false,
                "lshr",
            )?),
            BinaryOp::ArithShiftRight => {
                BasicValueEnum::IntValue(b.build_right_shift(lhs_int(), rhs_int(), true, "ashr")?)
            }
            BinaryOp::And => BasicValueEnum::IntValue(b.build_and(lhs_int(), rhs_int(), "and")?),
            BinaryOp::Or => BasicValueEnum::IntValue(b.build_or(lhs_int(), rhs_int(), "or")?),
            BinaryOp::Xor => BasicValueEnum::IntValue(b.build_xor(lhs_int(), rhs_int(), "xor")?),
            BinaryOp::FloatAdd => {
                BasicValueEnum::FloatValue(b.build_float_add(lhs_float(), rhs_float(), "fadd")?)
            }
            BinaryOp::FloatSub => {
                BasicValueEnum::FloatValue(b.build_float_sub(lhs_float(), rhs_float(), "fsub")?)
            }
            BinaryOp::FloatMul => {
                BasicValueEnum::FloatValue(b.build_float_mul(lhs_float(), rhs_float(), "fmul")?)
            }
            BinaryOp::FloatDiv => {
                BasicValueEnum::FloatValue(b.build_float_div(lhs_float(), rhs_float(), "fdiv")?)
            }
            BinaryOp::FloatRem => {
                BasicValueEnum::FloatValue(b.build_float_rem(lhs_float(), rhs_float(), "frem")?)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, RVSDGMod,
        func::{FnLinkageType, FnResult},
        lower_to_llvm::test_utils::test_utils::{jit_run_f32, jit_run_i32},
        types::{F32, I32},
        value::ConstValue,
    };

    fn build_binary_i32(op: BinaryOp, flags: ArithFlags, lhs: i32, rhs: i32) -> i32 {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(lhs);
            let b = rb.const_i32(rhs);
            let result = rb.binary(op, flags, a, b, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        jit_run_i32(&rvsdg, "test")
    }

    fn build_binary_f32(op: BinaryOp, lhs: f32, rhs: f32) -> f32 {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F32, ConstValue::f32_from_native(lhs));
            let b = rb.constant(F32, ConstValue::f32_from_native(rhs));
            let result = rb.binary(op, ArithFlags::default(), a, b, F32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        jit_run_f32(&rvsdg, "test")
    }

    // --- Integer arithmetic ---

    #[test]
    fn binary_add() {
        assert_eq!(
            build_binary_i32(BinaryOp::Add, ArithFlags::default(), 3, 7),
            10
        );
    }

    #[test]
    fn binary_sub() {
        assert_eq!(
            build_binary_i32(BinaryOp::Sub, ArithFlags::default(), 10, 4),
            6
        );
    }

    #[test]
    fn binary_sub_negative_result() {
        assert_eq!(
            build_binary_i32(BinaryOp::Sub, ArithFlags::default(), 3, 10),
            -7
        );
    }

    #[test]
    fn binary_mul() {
        assert_eq!(
            build_binary_i32(BinaryOp::Mul, ArithFlags::default(), 6, 7),
            42
        );
    }

    #[test]
    fn binary_signed_div() {
        assert_eq!(
            build_binary_i32(BinaryOp::SignedDiv, ArithFlags::default(), 20, 4),
            5
        );
    }

    #[test]
    fn binary_signed_div_negative() {
        assert_eq!(
            build_binary_i32(BinaryOp::SignedDiv, ArithFlags::default(), -20, 3),
            -6
        );
    }

    #[test]
    fn binary_unsigned_div() {
        assert_eq!(
            build_binary_i32(BinaryOp::UnsignedDiv, ArithFlags::default(), 20, 3),
            6
        );
    }

    #[test]
    fn binary_signed_rem() {
        assert_eq!(
            build_binary_i32(BinaryOp::SignedRem, ArithFlags::default(), 17, 5),
            2
        );
    }

    #[test]
    fn binary_signed_rem_negative() {
        assert_eq!(
            build_binary_i32(BinaryOp::SignedRem, ArithFlags::default(), -17, 5),
            -2
        );
    }

    #[test]
    fn binary_unsigned_rem() {
        assert_eq!(
            build_binary_i32(BinaryOp::UnsignedRem, ArithFlags::default(), 17, 5),
            2
        );
    }

    // --- Shifts ---

    #[test]
    fn binary_shift_left() {
        assert_eq!(
            build_binary_i32(BinaryOp::ShiftLeft, ArithFlags::default(), 1, 10),
            1024
        );
    }

    #[test]
    fn binary_logical_shift_right() {
        // 1024 >> 3 = 128
        assert_eq!(
            build_binary_i32(BinaryOp::LogicalShiftRight, ArithFlags::default(), 1024, 3),
            128
        );
    }

    #[test]
    fn binary_arith_shift_right() {
        // -16 >> 2 = -4 (sign-extended)
        assert_eq!(
            build_binary_i32(BinaryOp::ArithShiftRight, ArithFlags::default(), -16, 2),
            -4
        );
    }

    // --- Bitwise ---

    #[test]
    fn binary_and() {
        assert_eq!(
            build_binary_i32(BinaryOp::And, ArithFlags::default(), 0xFF, 0x0F),
            0x0F
        );
    }

    #[test]
    fn binary_or() {
        assert_eq!(
            build_binary_i32(BinaryOp::Or, ArithFlags::default(), 0xF0, 0x0F),
            0xFF
        );
    }

    #[test]
    fn binary_xor() {
        assert_eq!(
            build_binary_i32(BinaryOp::Xor, ArithFlags::default(), 0xFF, 0x0F),
            0xF0
        );
    }

    // --- Arithmetic flags ---

    #[test]
    fn binary_add_nsw() {
        let flags = ArithFlags {
            no_signed_wrap: true,
            ..ArithFlags::default()
        };
        assert_eq!(build_binary_i32(BinaryOp::Add, flags, 100, 200), 300);
    }

    #[test]
    fn binary_add_nuw() {
        let flags = ArithFlags {
            no_unsigned_wrap: true,
            ..ArithFlags::default()
        };
        assert_eq!(build_binary_i32(BinaryOp::Add, flags, 100, 200), 300);
    }

    #[test]
    fn binary_mul_nsw() {
        let flags = ArithFlags {
            no_signed_wrap: true,
            ..ArithFlags::default()
        };
        assert_eq!(build_binary_i32(BinaryOp::Mul, flags, 12, 11), 132);
    }

    #[test]
    fn binary_signed_div_exact() {
        let flags = ArithFlags {
            exact: true,
            ..ArithFlags::default()
        };
        assert_eq!(build_binary_i32(BinaryOp::SignedDiv, flags, 20, 4), 5);
    }

    // --- Float arithmetic ---

    #[test]
    fn binary_float_add() {
        assert_eq!(build_binary_f32(BinaryOp::FloatAdd, 1.5, 2.5), 4.0);
    }

    #[test]
    fn binary_float_sub() {
        assert_eq!(build_binary_f32(BinaryOp::FloatSub, 10.0, 3.5), 6.5);
    }

    #[test]
    fn binary_float_mul() {
        assert_eq!(build_binary_f32(BinaryOp::FloatMul, 3.0, 4.0), 12.0);
    }

    #[test]
    fn binary_float_div() {
        assert_eq!(build_binary_f32(BinaryOp::FloatDiv, 10.0, 4.0), 2.5);
    }

    #[test]
    fn binary_float_rem() {
        assert_eq!(build_binary_f32(BinaryOp::FloatRem, 10.0, 3.0), 1.0);
    }
}
