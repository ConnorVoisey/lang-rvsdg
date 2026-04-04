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
