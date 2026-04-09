use crate::rvsdg::{
    RVSDGMod, UnaryOp, ValueId,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
};
use inkwell::{
    builder::BuilderError,
    intrinsics::Intrinsic,
    types::BasicTypeEnum,
    values::{BasicMetadataValueEnum, BasicValueEnum, ValueKind},
};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_unary<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        op: UnaryOp,
        operand: ValueId,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let val = self.expect_value(llvm_builder, mapper, rvsdg_func, operand)?;
        let b = &llvm_builder.builder;
        let val_int = || val.into_int_value();
        let val_float = || val.into_float_value();

        Ok(match op {
            UnaryOp::BitNot => BasicValueEnum::IntValue(b.build_not(val_int(), "not")?),
            UnaryOp::FloatNeg => {
                BasicValueEnum::FloatValue(b.build_float_neg(val_float(), "fneg")?)
            }
            UnaryOp::CountLeadingZeros => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                let is_zero_poison = llvm_builder.context.bool_type().const_zero();
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ctlz",
                    &[int_type],
                    &[int_val.into(), is_zero_poison.into()],
                    "ctlz",
                )
            }
            UnaryOp::CountTrailingZeros => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                let is_zero_poison = llvm_builder.context.bool_type().const_zero();
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.cttz",
                    &[int_type],
                    &[int_val.into(), is_zero_poison.into()],
                    "cttz",
                )
            }
            UnaryOp::CountOnes => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ctpop",
                    &[int_type],
                    &[int_val.into()],
                    "ctpop",
                )
            }
            UnaryOp::ByteSwap => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.bswap",
                    &[int_type],
                    &[int_val.into()],
                    "bswap",
                )
            }
            UnaryOp::BitReverse => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.bitreverse",
                    &[int_type],
                    &[int_val.into()],
                    "bitrev",
                )
            }
            UnaryOp::FloatAbs => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.fabs",
                    &[float_type],
                    &[float_val.into()],
                    "fabs",
                )
            }
            UnaryOp::FloatFloor => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.floor",
                    &[float_type],
                    &[float_val.into()],
                    "floor",
                )
            }
            UnaryOp::FloatCeil => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ceil",
                    &[float_type],
                    &[float_val.into()],
                    "ceil",
                )
            }
            UnaryOp::FloatRound => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.round",
                    &[float_type],
                    &[float_val.into()],
                    "round",
                )
            }
            UnaryOp::FloatSqrt => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.sqrt",
                    &[float_type],
                    &[float_val.into()],
                    "sqrt",
                )
            }
        })
    }

    pub(crate) fn call_overloaded_intrinsic<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        intrinsic_name: &str,
        param_types: &[BasicTypeEnum<'ctx>],
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let intrinsic = Intrinsic::find(intrinsic_name)
            .unwrap_or_else(|| panic!("intrinsic {intrinsic_name} not found"));
        let func = intrinsic
            .get_declaration(llvm_builder.module, param_types)
            .unwrap_or_else(|| panic!("failed to get declaration for {intrinsic_name}"));
        match llvm_builder
            .builder
            .build_call(func, args, name)
            .expect("failed to build intrinsic call")
            .try_as_basic_value()
        {
            ValueKind::Basic(val) => val,
            ValueKind::Instruction(_) => {
                panic!("intrinsic {intrinsic_name} returned void, expected a value")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, RVSDGMod, UnaryOp,
        func::{FnLinkageType, FnResult},
        lower_to_llvm::test_utils::test_utils::{jit_run_f32, jit_run_i32},
        types::{F32, I32},
        value::ConstValue,
    };

    fn build_unary_i32(op: UnaryOp, input: i32) -> i32 {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(input);
            let result = rb.unary(op, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        jit_run_i32(&rvsdg, "test")
    }

    fn build_unary_f32(op: UnaryOp, input: f32) -> f32 {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(F32, ConstValue::f32_from_native(input));
            let result = rb.unary(op, v, F32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        jit_run_f32(&rvsdg, "test")
    }

    // --- BitNot ---

    #[test]
    fn unary_bit_not() {
        assert_eq!(build_unary_i32(UnaryOp::BitNot, 0), -1);
    }

    #[test]
    fn unary_bit_not_pattern() {
        // ~0x0F0F0F0F = 0xF0F0F0F0
        assert_eq!(
            build_unary_i32(UnaryOp::BitNot, 0x0F0F0F0F),
            0xF0F0F0F0_u32 as i32
        );
    }

    // --- CountLeadingZeros ---

    #[test]
    fn unary_clz_one() {
        // 1 = 0x00000001, 31 leading zeros
        assert_eq!(build_unary_i32(UnaryOp::CountLeadingZeros, 1), 31);
    }

    #[test]
    fn unary_clz_high_bit() {
        // i32::MIN = 0x80000000, 0 leading zeros
        assert_eq!(build_unary_i32(UnaryOp::CountLeadingZeros, i32::MIN), 0);
    }

    #[test]
    fn unary_clz_zero() {
        // 0 has 32 leading zeros (is_zero_poison = false)
        assert_eq!(build_unary_i32(UnaryOp::CountLeadingZeros, 0), 32);
    }

    // --- CountTrailingZeros ---

    #[test]
    fn unary_ctz_one() {
        // 1 = 0b...0001, 0 trailing zeros
        assert_eq!(build_unary_i32(UnaryOp::CountTrailingZeros, 1), 0);
    }

    #[test]
    fn unary_ctz_power_of_two() {
        // 16 = 0b10000, 4 trailing zeros
        assert_eq!(build_unary_i32(UnaryOp::CountTrailingZeros, 16), 4);
    }

    #[test]
    fn unary_ctz_zero() {
        // 0 has 32 trailing zeros (is_zero_poison = false)
        assert_eq!(build_unary_i32(UnaryOp::CountTrailingZeros, 0), 32);
    }

    // --- CountOnes (popcount) ---

    #[test]
    fn unary_popcount_zero() {
        assert_eq!(build_unary_i32(UnaryOp::CountOnes, 0), 0);
    }

    #[test]
    fn unary_popcount_all_ones() {
        assert_eq!(build_unary_i32(UnaryOp::CountOnes, -1), 32);
    }

    #[test]
    fn unary_popcount_pattern() {
        // 0b1010_1010 = 0xAA, 4 ones
        assert_eq!(build_unary_i32(UnaryOp::CountOnes, 0xAA), 4);
    }

    // --- ByteSwap ---

    #[test]
    fn unary_bswap() {
        // 0x01020304 -> 0x04030201
        assert_eq!(build_unary_i32(UnaryOp::ByteSwap, 0x01020304), 0x04030201);
    }

    #[test]
    fn unary_bswap_roundtrip() {
        // bswap(bswap(x)) == x
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(0xDEADBEEF_u32 as i32);
            let swapped = rb.unary(UnaryOp::ByteSwap, v, I32);
            let result = rb.unary(UnaryOp::ByteSwap, swapped, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0xDEADBEEF_u32 as i32);
    }

    // --- BitReverse ---

    #[test]
    fn unary_bitreverse_one() {
        // bit 0 set -> bit 31 set = 0x80000000 = i32::MIN
        assert_eq!(build_unary_i32(UnaryOp::BitReverse, 1), i32::MIN);
    }

    #[test]
    fn unary_bitreverse_roundtrip() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(0x12345678);
            let rev = rb.unary(UnaryOp::BitReverse, v, I32);
            let result = rb.unary(UnaryOp::BitReverse, rev, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0x12345678);
    }

    // --- FloatNeg ---

    #[test]
    fn unary_float_neg() {
        assert_eq!(build_unary_f32(UnaryOp::FloatNeg, 3.5), -3.5);
    }

    #[test]
    fn unary_float_neg_negative() {
        assert_eq!(build_unary_f32(UnaryOp::FloatNeg, -7.0), 7.0);
    }

    // --- FloatAbs ---

    #[test]
    fn unary_float_abs_positive() {
        assert_eq!(build_unary_f32(UnaryOp::FloatAbs, 5.0), 5.0);
    }

    #[test]
    fn unary_float_abs_negative() {
        assert_eq!(build_unary_f32(UnaryOp::FloatAbs, -5.0), 5.0);
    }

    // --- FloatFloor ---

    #[test]
    fn unary_float_floor() {
        assert_eq!(build_unary_f32(UnaryOp::FloatFloor, 3.7), 3.0);
    }

    #[test]
    fn unary_float_floor_negative() {
        assert_eq!(build_unary_f32(UnaryOp::FloatFloor, -3.2), -4.0);
    }

    // --- FloatCeil ---

    #[test]
    fn unary_float_ceil() {
        assert_eq!(build_unary_f32(UnaryOp::FloatCeil, 3.2), 4.0);
    }

    #[test]
    fn unary_float_ceil_negative() {
        assert_eq!(build_unary_f32(UnaryOp::FloatCeil, -3.7), -3.0);
    }

    // --- FloatRound ---

    #[test]
    fn unary_float_round_up() {
        assert_eq!(build_unary_f32(UnaryOp::FloatRound, 3.5), 4.0);
    }

    #[test]
    fn unary_float_round_down() {
        assert_eq!(build_unary_f32(UnaryOp::FloatRound, 3.4), 3.0);
    }

    // --- FloatSqrt ---

    #[test]
    fn unary_float_sqrt() {
        assert_eq!(build_unary_f32(UnaryOp::FloatSqrt, 25.0), 5.0);
    }

    #[test]
    fn unary_float_sqrt_fractional() {
        assert_eq!(build_unary_f32(UnaryOp::FloatSqrt, 2.25), 1.5);
    }

    // --- Composed: abs(neg(x)) == abs(x) ---

    #[test]
    fn unary_abs_neg_composition() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(F32, ConstValue::f32_from_native(42.0));
            let neg = rb.unary(UnaryOp::FloatNeg, v, F32);
            let result = rb.unary(UnaryOp::FloatAbs, neg, F32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 42.0);
    }
}
