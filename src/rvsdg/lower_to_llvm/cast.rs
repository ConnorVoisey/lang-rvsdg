use crate::rvsdg::{
    CastOp, RVSDGMod, Value, ValueId,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
};
use inkwell::{builder::BuilderError, values::BasicValueEnum};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_cast<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        op: CastOp,
        operand: ValueId,
        value: &Value,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let src = self.expect_value(llvm_builder, mapper, rvsdg_func, operand)?;
        let dst_type = self.type_to_basic_type_llvm(llvm_builder.context, value.ty);
        let b = &llvm_builder.builder;
        Ok(match op {
            CastOp::SignExtend => BasicValueEnum::IntValue(b.build_int_s_extend(
                src.into_int_value(),
                dst_type.into_int_type(),
                "sext",
            )?),
            CastOp::ZeroExtend => BasicValueEnum::IntValue(b.build_int_z_extend(
                src.into_int_value(),
                dst_type.into_int_type(),
                "zext",
            )?),
            CastOp::Truncate => BasicValueEnum::IntValue(b.build_int_truncate(
                src.into_int_value(),
                dst_type.into_int_type(),
                "trunc",
            )?),
            CastOp::FloatExtend => BasicValueEnum::FloatValue(b.build_float_ext(
                src.into_float_value(),
                dst_type.into_float_type(),
                "fpext",
            )?),
            CastOp::FloatTruncate => BasicValueEnum::FloatValue(b.build_float_trunc(
                src.into_float_value(),
                dst_type.into_float_type(),
                "fptrunc",
            )?),
            CastOp::SignedToFloat => BasicValueEnum::FloatValue(b.build_signed_int_to_float(
                src.into_int_value(),
                dst_type.into_float_type(),
                "sitofp",
            )?),
            CastOp::UnsignedToFloat => BasicValueEnum::FloatValue(b.build_unsigned_int_to_float(
                src.into_int_value(),
                dst_type.into_float_type(),
                "uitofp",
            )?),
            CastOp::FloatToSigned => BasicValueEnum::IntValue(b.build_float_to_signed_int(
                src.into_float_value(),
                dst_type.into_int_type(),
                "fptosi",
            )?),
            CastOp::FloatToUnsigned => BasicValueEnum::IntValue(b.build_float_to_unsigned_int(
                src.into_float_value(),
                dst_type.into_int_type(),
                "fptoui",
            )?),
            CastOp::PtrToInt => BasicValueEnum::IntValue(b.build_ptr_to_int(
                src.into_pointer_value(),
                dst_type.into_int_type(),
                "ptrtoint",
            )?),
            CastOp::IntToPtr => BasicValueEnum::PointerValue(b.build_int_to_ptr(
                src.into_int_value(),
                dst_type.into_pointer_type(),
                "inttoptr",
            )?),
            CastOp::Bitcast => b.build_bit_cast(src, dst_type, "bitcast")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, CastOp, ICmpPred, Linkage, RVSDGMod,
        func::FnResult,
        lower_to_llvm::test_utils::test_utils::{
            jit_run_f32, jit_run_f64, jit_run_i32, jit_run_i64,
        },
        types::{F32, F64, I8, I16, I32, I64, PtrType, TypeRef},
        value::ConstValue,
    };

    // --- Integer widening ---

    #[test]
    fn cast_sign_extend_i8_to_i32() {
        // -1 as i8 sign-extended to i32 should be -1
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(I8, ConstValue::Int(-1i8 as i64));
            let result = rb.cast(CastOp::SignExtend, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), -1);
    }

    #[test]
    fn cast_sign_extend_positive() {
        // 100 as i8 sign-extended to i32 should be 100
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(I8, ConstValue::Int(100));
            let result = rb.cast(CastOp::SignExtend, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 100);
    }

    #[test]
    fn cast_zero_extend_i8_to_i32() {
        // 0xFF as i8 zero-extended to i32 should be 255, not -1
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(I8, ConstValue::Int(0xFF));
            let result = rb.cast(CastOp::ZeroExtend, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 255);
    }

    #[test]
    fn cast_zero_extend_i16_to_i64() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I64], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(I16, ConstValue::Int(50000));
            let result = rb.cast(CastOp::ZeroExtend, v, I64);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i64(&rvsdg, "test"), 50000);
    }

    // --- Integer narrowing ---

    #[test]
    fn cast_truncate_i32_to_i8() {
        // 0x1FF truncated to i8 should be 0xFF = -1 as signed i8
        // Verify via zero-extending back to i32: should be 255
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(0x1FF);
            let truncated = rb.cast(CastOp::Truncate, v, I8);
            let result = rb.cast(CastOp::ZeroExtend, truncated, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 255);
    }

    #[test]
    fn cast_truncate_i64_to_i32() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i64(0x1_0000_002A); // lower 32 bits = 42
            let result = rb.cast(CastOp::Truncate, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    // --- Float widening/narrowing ---

    #[test]
    fn cast_float_extend_f32_to_f64() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F64], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(F32, ConstValue::f32_from_native(3.5));
            let result = rb.cast(CastOp::FloatExtend, v, F64);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_f64(&rvsdg, "test"), 3.5);
    }

    #[test]
    fn cast_float_truncate_f64_to_f32() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(F64, ConstValue::f64_from_native(2.5));
            let result = rb.cast(CastOp::FloatTruncate, v, F32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 2.5);
    }

    // --- Int <-> Float conversions ---

    #[test]
    fn cast_signed_to_float() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(-42);
            let result = rb.cast(CastOp::SignedToFloat, v, F32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), -42.0);
    }

    #[test]
    fn cast_unsigned_to_float() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            // 3_000_000_000 doesn't fit in signed i32 but is a valid u32
            let v = rb.const_i32(3_000_000_000u32 as i32);
            let result = rb.cast(CastOp::UnsignedToFloat, v, F32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 3_000_000_000.0f32);
    }

    #[test]
    fn cast_float_to_signed() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(F32, ConstValue::f32_from_native(-7.9));
            let result = rb.cast(CastOp::FloatToSigned, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        // truncates toward zero
        assert_eq!(jit_run_i32(&rvsdg, "test"), -7);
    }

    #[test]
    fn cast_float_to_unsigned() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(F32, ConstValue::f32_from_native(42.7));
            let result = rb.cast(CastOp::FloatToUnsigned, v, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    // --- Pointer conversions ---

    #[test]
    fn cast_ptr_to_int_roundtrip() {
        // alloca a pointer, convert to int, convert back, store through it, load
        use crate::rvsdg::GlobalInit;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty_id = rvsdg.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        });
        let ptr_ty = TypeRef::Ptr(ptr_ty_id);
        let init = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let global_id = rvsdg.define_global(
            String::from("val"),
            I32,
            GlobalInit::Init(init),
            false,
            Linkage::Internal,
        );
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            // ptr -> int -> ptr roundtrip
            let as_int = rb.cast(CastOp::PtrToInt, ptr, I64);
            let as_ptr = rb.cast(CastOp::IntToPtr, as_int, ptr_ty);
            // store 99 through the roundtripped pointer
            let ninety_nine = rb.const_i32(99);
            let s1 = rb.store(state, as_ptr, ninety_nine, None, false);
            // load from original pointer
            let loaded = rb.load(s1, ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 99);
    }

    // --- Chained casts ---

    #[test]
    fn cast_chain_i32_to_f64_to_f32_to_i32() {
        // 42 -> f64 -> f32 -> i32 should survive the round trip
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(42);
            let as_f64 = rb.cast(CastOp::SignedToFloat, v, F64);
            let as_f32 = rb.cast(CastOp::FloatTruncate, as_f64, F32);
            let result = rb.cast(CastOp::FloatToSigned, as_f32, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn cast_sext_then_truncate_identity() {
        // i8 -> sext i32 -> trunc i8 -> zext i32: for positive values, should be identity
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.constant(I8, ConstValue::Int(77));
            let wide = rb.cast(CastOp::SignExtend, v, I32);
            let narrow = rb.cast(CastOp::Truncate, wide, I8);
            let result = rb.cast(CastOp::ZeroExtend, narrow, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 77);
    }
}
