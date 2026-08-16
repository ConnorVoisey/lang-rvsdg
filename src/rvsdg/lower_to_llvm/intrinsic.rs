use crate::rvsdg::{IntrinsicOp, ValueId, ValuesSpan, lower_to_llvm::FunctionLowerer};
use color_eyre::eyre::{bail, eyre};
use inkwell::{
    intrinsics::Intrinsic,
    types::BasicTypeEnum,
    values::{BasicMetadataValueEnum, BasicValueEnum, ValueKind},
};

impl<'m, 'a, 'ctx> FunctionLowerer<'m, 'a, 'ctx> {
    pub(crate) fn lower_intrinsic(
        &mut self,
        op: IntrinsicOp,
        args: ValuesSpan,
        value_id: ValueId,
    ) -> color_eyre::Result<()> {
        let graph = self.graph;
        let arg_vals: Vec<BasicValueEnum<'ctx>> = graph
            .value_pool
            .get(args)
            .iter()
            .map(|&id| self.expect_value(id))
            .collect::<color_eyre::Result<_>>()?;

        match op {
            // Void intrinsics (state-only, no Project)
            IntrinsicOp::MemCopy => {
                self.call_void_intrinsic("llvm.memcpy", &arg_vals)?;
            }
            IntrinsicOp::MemMove => {
                self.call_void_intrinsic("llvm.memmove", &arg_vals)?;
            }
            IntrinsicOp::MemSet => {
                self.call_void_intrinsic("llvm.memset", &arg_vals)?;
            }
            IntrinsicOp::LifetimeStart | IntrinsicOp::LifetimeEnd => {
                // Optimizer hints -- emit nothing for now
            }
            IntrinsicOp::Unreachable => {
                self.builder.build_unreachable()?;
            }
            IntrinsicOp::Expect => {
                // llvm.expect.i1(condition, expected) -> condition
                // Just pass through the condition value as the result
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, arg_vals[0]);
            }

            // Single-result intrinsics (Project{0} for the value)
            IntrinsicOp::IntAbs => {
                let int_val = arg_vals[0].into_int_value();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                let is_poison = self.mod_lower.context.bool_type().const_zero();
                let result = self.call_overloaded_intrinsic(
                    "llvm.abs",
                    &[int_type],
                    &[int_val.into(), is_poison.into()],
                    "abs",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::FloatFma => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.fma",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into(), arg_vals[2].into()],
                    "fma",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::FloatMin => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.minnum",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "fmin",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::FloatMax => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.maxnum",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "fmax",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::FloatCopySign => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.copysign",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "copysign",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::SignedAddSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.sadd.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "sadd.sat",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedAddSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.uadd.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "uadd.sat",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::SignedSubSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.ssub.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "ssub.sat",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedSubSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.usub.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "usub.sat",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::SignedMin => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.smin",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "smin",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::SignedMax => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.smax",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "smax",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedMin => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.umin",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "umin",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedMax => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    "llvm.umax",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "umax",
                )?;
                let project_id = graph.projection_of(value_id, 0);
                self.set_val(project_id, result);
            }

            // Two-result overflow intrinsics (Project{0} = result, Project{1} = overflow flag)
            IntrinsicOp::SignedAddOverflow => {
                self.lower_overflow_intrinsic("llvm.sadd.with.overflow", &arg_vals, value_id)?;
            }
            IntrinsicOp::UnsignedAddOverflow => {
                self.lower_overflow_intrinsic("llvm.uadd.with.overflow", &arg_vals, value_id)?;
            }
            IntrinsicOp::SignedSubOverflow => {
                self.lower_overflow_intrinsic("llvm.ssub.with.overflow", &arg_vals, value_id)?;
            }
            IntrinsicOp::UnsignedSubOverflow => {
                self.lower_overflow_intrinsic("llvm.usub.with.overflow", &arg_vals, value_id)?;
            }
            IntrinsicOp::SignedMulOverflow => {
                self.lower_overflow_intrinsic("llvm.smul.with.overflow", &arg_vals, value_id)?;
            }
            IntrinsicOp::UnsignedMulOverflow => {
                self.lower_overflow_intrinsic("llvm.umul.with.overflow", &arg_vals, value_id)?;
            }
        }

        Ok(())
    }

    fn call_void_intrinsic(
        &self,
        name: &str,
        args: &[BasicValueEnum<'ctx>],
    ) -> color_eyre::Result<()> {
        let param_types: Vec<BasicTypeEnum<'ctx>> = args.iter().map(|a| a.get_type()).collect();
        let intrinsic =
            Intrinsic::find(name).ok_or_else(|| eyre!("intrinsic `{name}` not found"))?;
        let func = intrinsic
            .get_declaration(self.mod_lower.module, &param_types)
            .ok_or_else(|| eyre!("failed to get declaration for intrinsic `{name}`"))?;
        let meta_args: Vec<BasicMetadataValueEnum<'ctx>> = args.iter().map(|&a| a.into()).collect();
        self.builder.build_call(func, &meta_args, name)?;
        Ok(())
    }

    fn lower_overflow_intrinsic(
        &mut self,
        name: &str,
        args: &[BasicValueEnum<'ctx>],
        value_id: ValueId,
    ) -> color_eyre::Result<()> {
        let int_type = BasicTypeEnum::IntType(args[0].into_int_value().get_type());
        let intrinsic =
            Intrinsic::find(name).ok_or_else(|| eyre!("intrinsic `{name}` not found"))?;
        let func = intrinsic
            .get_declaration(self.mod_lower.module, &[int_type])
            .ok_or_else(|| eyre!("failed to get declaration for intrinsic `{name}`"))?;
        let meta_args: Vec<BasicMetadataValueEnum<'ctx>> = args.iter().map(|&a| a.into()).collect();
        let call_result = self
            .builder
            .build_call(func, &meta_args, name)?
            .try_as_basic_value();
        match call_result {
            ValueKind::Basic(struct_val) => {
                // The overflow intrinsic returns {iN, i1}
                let sv = struct_val.into_struct_value();
                let result = self.builder.build_extract_value(sv, 0, "result")?;
                let overflow = self.builder.build_extract_value(sv, 1, "overflow")?;
                let project_0 = self.graph.projection_of(value_id, 0);
                let project_1 = self.graph.projection_of(value_id, 1);
                self.set_val(project_0, result);
                self.set_val(project_1, overflow);
            }
            ValueKind::Instruction(_) => {
                bail!("overflow intrinsic `{name}` unexpectedly returned void")
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        IntrinsicOp, Linkage, RVSDGMod,
        lower_to_llvm::test_utils::test_utils::{jit_run_f32, jit_run_i32},
        types::{F32, I32},
        value::ConstValue,
    };

    // --- IntAbs ---

    #[test]
    fn intrinsic_int_abs_positive() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let v = rb.const_i32(42);
                let res = rb.intrinsic(IntrinsicOp::IntAbs, &[v], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn intrinsic_int_abs_negative() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let v = rb.const_i32(-42);
                let res = rb.intrinsic(IntrinsicOp::IntAbs, &[v], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    // --- Min / Max ---

    #[test]
    fn intrinsic_signed_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(10);
                let b = rb.const_i32(-5);
                let res = rb.intrinsic(IntrinsicOp::SignedMin, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), -5);
    }

    #[test]
    fn intrinsic_signed_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(10);
                let b = rb.const_i32(-5);
                let res = rb.intrinsic(IntrinsicOp::SignedMax, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn intrinsic_unsigned_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(3);
                let b = rb.const_i32(7);
                let res = rb.intrinsic(IntrinsicOp::UnsignedMin, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 3);
    }

    #[test]
    fn intrinsic_unsigned_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(3);
                let b = rb.const_i32(7);
                let res = rb.intrinsic(IntrinsicOp::UnsignedMax, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 7);
    }

    // --- Saturating arithmetic ---

    #[test]
    fn intrinsic_sadd_sat_no_overflow() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(100);
                let b = rb.const_i32(200);
                let res = rb.intrinsic(IntrinsicOp::SignedAddSaturate, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 300);
    }

    #[test]
    fn intrinsic_sadd_sat_overflow() {
        // i32::MAX + 1 should saturate to i32::MAX
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(i32::MAX);
                let b = rb.const_i32(1);
                let res = rb.intrinsic(IntrinsicOp::SignedAddSaturate, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), i32::MAX);
    }

    #[test]
    fn intrinsic_usub_sat_clamp_to_zero() {
        // 3 - 10 unsigned saturates to 0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(3);
                let b = rb.const_i32(10);
                let res = rb.intrinsic(IntrinsicOp::UnsignedSubSaturate, &[a, b], I32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    // --- Float intrinsics ---

    #[test]
    fn intrinsic_float_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.constant(F32, ConstValue::f32_from_native(3.5));
                let b = rb.constant(F32, ConstValue::f32_from_native(1.5));
                let res = rb.intrinsic(IntrinsicOp::FloatMin, &[a, b], F32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_f32(&rvsdg, "test"), 1.5);
    }

    #[test]
    fn intrinsic_float_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.constant(F32, ConstValue::f32_from_native(3.5));
                let b = rb.constant(F32, ConstValue::f32_from_native(1.5));
                let res = rb.intrinsic(IntrinsicOp::FloatMax, &[a, b], F32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_f32(&rvsdg, "test"), 3.5);
    }

    #[test]
    fn intrinsic_float_fma() {
        // fma(2.0, 3.0, 4.0) = 2*3 + 4 = 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.constant(F32, ConstValue::f32_from_native(2.0));
                let b = rb.constant(F32, ConstValue::f32_from_native(3.0));
                let c = rb.constant(F32, ConstValue::f32_from_native(4.0));
                let res = rb.intrinsic(IntrinsicOp::FloatFma, &[a, b, c], F32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_f32(&rvsdg, "test"), 10.0);
    }

    #[test]
    fn intrinsic_float_copysign() {
        // copysign(5.0, -1.0) = -5.0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let mag = rb.constant(F32, ConstValue::f32_from_native(5.0));
                let sign = rb.constant(F32, ConstValue::f32_from_native(-1.0));
                let res = rb.intrinsic(IntrinsicOp::FloatCopySign, &[mag, sign], F32);
                Ok(vec![res])
            })
            .unwrap();
        assert_eq!(jit_run_f32(&rvsdg, "test"), -5.0);
    }

    // --- Overflow-checked arithmetic ---

    #[test]
    fn intrinsic_sadd_overflow_no_overflow() {
        // 100 + 200 = 300, no overflow => return result
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(100);
                let b = rb.const_i32(200);
                let res = rb.intrinsic_overflow(IntrinsicOp::SignedAddOverflow, &[a, b], I32);
                Ok(vec![res.value])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 300);
    }

    #[test]
    fn intrinsic_sadd_overflow_flag() {
        // i32::MAX + 1 overflows => overflow flag is true (1)
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(i32::MAX);
                let b = rb.const_i32(1);
                let res = rb.intrinsic_overflow(IntrinsicOp::SignedAddOverflow, &[a, b], I32);
                // overflow flag is i1, zero-extend to i32 to return it
                let flag = rb.cast(crate::rvsdg::CastOp::ZeroExtend, res.overflow, I32);
                Ok(vec![flag])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn intrinsic_sadd_overflow_no_flag() {
        // 1 + 2 doesn't overflow => flag is 0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(1);
                let b = rb.const_i32(2);
                let res = rb.intrinsic_overflow(IntrinsicOp::SignedAddOverflow, &[a, b], I32);
                let flag = rb.cast(crate::rvsdg::CastOp::ZeroExtend, res.overflow, I32);
                Ok(vec![flag])
            })
            .unwrap();
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    #[test]
    fn intrinsic_umul_overflow() {
        // Large unsigned multiply that overflows: 0x80000000 * 2 overflows u32
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg
            .define_fn(func_id, |rb| {
                let a = rb.const_i32(0x40000000);
                let b = rb.const_i32(4);
                let res = rb.intrinsic_overflow(IntrinsicOp::UnsignedMulOverflow, &[a, b], I32);
                let flag = rb.cast(crate::rvsdg::CastOp::ZeroExtend, res.overflow, I32);
                Ok(vec![flag])
            })
            .unwrap();
        // 0x40000000 * 4 = 0x100000000 which overflows u32
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }
}
