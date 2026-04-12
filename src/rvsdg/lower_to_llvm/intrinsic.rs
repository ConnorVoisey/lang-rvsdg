use crate::rvsdg::{
    IntrinsicOp, RVSDGMod, ValueId, ValuesSpan,
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
    pub(crate) fn lower_intrinsic<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        op: IntrinsicOp,
        args: ValuesSpan,
        value_id: ValueId,
    ) -> Result<(), BuilderError> {
        let arg_ids = self.value_pool.get(args).to_vec();
        let arg_vals: Vec<BasicValueEnum<'ctx>> = arg_ids
            .iter()
            .map(|&id| {
                self.expect_value(llvm_builder, mapper, rvsdg_func, id)
                    .expect("failed to lower intrinsic argument")
            })
            .collect();

        match op {
            // Void intrinsics (state-only, no Project)
            IntrinsicOp::MemCopy => {
                self.call_void_intrinsic(llvm_builder, "llvm.memcpy", &arg_vals)?;
            }
            IntrinsicOp::MemMove => {
                self.call_void_intrinsic(llvm_builder, "llvm.memmove", &arg_vals)?;
            }
            IntrinsicOp::MemSet => {
                self.call_void_intrinsic(llvm_builder, "llvm.memset", &arg_vals)?;
            }
            IntrinsicOp::LifetimeStart | IntrinsicOp::LifetimeEnd => {
                // Optimizer hints — emit nothing for now
            }
            IntrinsicOp::Unreachable => {
                llvm_builder.builder.build_unreachable()?;
            }
            IntrinsicOp::Expect => {
                // llvm.expect.i1(condition, expected) -> condition
                // Just pass through the condition value as the result
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, arg_vals[0]);
            }

            // Single-result intrinsics (Project{0} for the value)
            IntrinsicOp::IntAbs => {
                let int_val = arg_vals[0].into_int_value();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                let is_poison = llvm_builder.context.bool_type().const_zero();
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.abs",
                    &[int_type],
                    &[int_val.into(), is_poison.into()],
                    "abs",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::FloatFma => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.fma",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into(), arg_vals[2].into()],
                    "fma",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::FloatMin => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.minnum",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "fmin",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::FloatMax => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.maxnum",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "fmax",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::FloatCopySign => {
                let float_type =
                    BasicTypeEnum::FloatType(arg_vals[0].into_float_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.copysign",
                    &[float_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "copysign",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::SignedAddSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.sadd.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "sadd.sat",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedAddSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.uadd.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "uadd.sat",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::SignedSubSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ssub.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "ssub.sat",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedSubSaturate => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.usub.sat",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "usub.sat",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::SignedMin => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.smin",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "smin",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::SignedMax => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.smax",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "smax",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedMin => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.umin",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "umin",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }
            IntrinsicOp::UnsignedMax => {
                let int_type = BasicTypeEnum::IntType(arg_vals[0].into_int_value().get_type());
                let result = self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.umax",
                    &[int_type],
                    &[arg_vals[0].into(), arg_vals[1].into()],
                    "umax",
                );
                let project_id = ValueId(value_id.0 + 1);
                mapper.set_val(project_id, result);
            }

            // Two-result overflow intrinsics (Project{0} = result, Project{1} = overflow flag)
            IntrinsicOp::SignedAddOverflow => {
                self.lower_overflow_intrinsic(
                    llvm_builder,
                    mapper,
                    "llvm.sadd.with.overflow",
                    &arg_vals,
                    value_id,
                );
            }
            IntrinsicOp::UnsignedAddOverflow => {
                self.lower_overflow_intrinsic(
                    llvm_builder,
                    mapper,
                    "llvm.uadd.with.overflow",
                    &arg_vals,
                    value_id,
                );
            }
            IntrinsicOp::SignedSubOverflow => {
                self.lower_overflow_intrinsic(
                    llvm_builder,
                    mapper,
                    "llvm.ssub.with.overflow",
                    &arg_vals,
                    value_id,
                );
            }
            IntrinsicOp::UnsignedSubOverflow => {
                self.lower_overflow_intrinsic(
                    llvm_builder,
                    mapper,
                    "llvm.usub.with.overflow",
                    &arg_vals,
                    value_id,
                );
            }
            IntrinsicOp::SignedMulOverflow => {
                self.lower_overflow_intrinsic(
                    llvm_builder,
                    mapper,
                    "llvm.smul.with.overflow",
                    &arg_vals,
                    value_id,
                );
            }
            IntrinsicOp::UnsignedMulOverflow => {
                self.lower_overflow_intrinsic(
                    llvm_builder,
                    mapper,
                    "llvm.umul.with.overflow",
                    &arg_vals,
                    value_id,
                );
            }
        }

        Ok(())
    }

    fn call_void_intrinsic<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        name: &str,
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<(), BuilderError> {
        let param_types: Vec<BasicTypeEnum<'ctx>> = args.iter().map(|a| a.get_type()).collect();
        let intrinsic =
            Intrinsic::find(name).unwrap_or_else(|| panic!("intrinsic {name} not found"));
        let func = intrinsic
            .get_declaration(llvm_builder.module, &param_types)
            .unwrap_or_else(|| panic!("failed to get declaration for {name}"));
        let meta_args: Vec<BasicMetadataValueEnum<'ctx>> = args.iter().map(|&a| a.into()).collect();
        llvm_builder
            .builder
            .build_call(func, &meta_args, name)
            .expect("failed to build intrinsic call");
        Ok(())
    }

    fn lower_overflow_intrinsic<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        name: &str,
        args: &[BasicValueEnum<'ctx>],
        value_id: ValueId,
    ) {
        let int_type = BasicTypeEnum::IntType(args[0].into_int_value().get_type());
        let intrinsic =
            Intrinsic::find(name).unwrap_or_else(|| panic!("intrinsic {name} not found"));
        let func = intrinsic
            .get_declaration(llvm_builder.module, &[int_type])
            .unwrap_or_else(|| panic!("failed to get declaration for {name}"));
        let meta_args: Vec<BasicMetadataValueEnum<'ctx>> = args.iter().map(|&a| a.into()).collect();
        let call_result = llvm_builder
            .builder
            .build_call(func, &meta_args, name)
            .expect("failed to build intrinsic call")
            .try_as_basic_value();
        match call_result {
            ValueKind::Basic(struct_val) => {
                // The overflow intrinsic returns {iN, i1}
                let sv = struct_val.into_struct_value();
                let result = llvm_builder
                    .builder
                    .build_extract_value(sv, 0, "result")
                    .expect("failed to extract overflow result");
                let overflow = llvm_builder
                    .builder
                    .build_extract_value(sv, 1, "overflow")
                    .expect("failed to extract overflow flag");
                let project_0 = ValueId(value_id.0 + 1);
                let project_1 = ValueId(value_id.0 + 2);
                mapper.set_val(project_0, result);
                mapper.set_val(project_1, overflow);
            }
            ValueKind::Instruction(_) => panic!("overflow intrinsic returned void"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        IntrinsicOp, Linkage, RVSDGMod,
        func::FnResult,
        lower_to_llvm::test_utils::test_utils::{jit_run_f32, jit_run_i32},
        types::{F32, I32},
        value::ConstValue,
    };

    // --- IntAbs ---

    #[test]
    fn intrinsic_int_abs_positive() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(42);
            let res = rb.intrinsic(IntrinsicOp::IntAbs, state, &[v], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn intrinsic_int_abs_negative() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(-42);
            let res = rb.intrinsic(IntrinsicOp::IntAbs, state, &[v], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    // --- Min / Max ---

    #[test]
    fn intrinsic_signed_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(10);
            let b = rb.const_i32(-5);
            let res = rb.intrinsic(IntrinsicOp::SignedMin, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), -5);
    }

    #[test]
    fn intrinsic_signed_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(10);
            let b = rb.const_i32(-5);
            let res = rb.intrinsic(IntrinsicOp::SignedMax, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn intrinsic_unsigned_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(3);
            let b = rb.const_i32(7);
            let res = rb.intrinsic(IntrinsicOp::UnsignedMin, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 3);
    }

    #[test]
    fn intrinsic_unsigned_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(3);
            let b = rb.const_i32(7);
            let res = rb.intrinsic(IntrinsicOp::UnsignedMax, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 7);
    }

    // --- Saturating arithmetic ---

    #[test]
    fn intrinsic_sadd_sat_no_overflow() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(100);
            let b = rb.const_i32(200);
            let res = rb.intrinsic(IntrinsicOp::SignedAddSaturate, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 300);
    }

    #[test]
    fn intrinsic_sadd_sat_overflow() {
        // i32::MAX + 1 should saturate to i32::MAX
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(i32::MAX);
            let b = rb.const_i32(1);
            let res = rb.intrinsic(IntrinsicOp::SignedAddSaturate, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), i32::MAX);
    }

    #[test]
    fn intrinsic_usub_sat_clamp_to_zero() {
        // 3 - 10 unsigned saturates to 0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(3);
            let b = rb.const_i32(10);
            let res = rb.intrinsic(IntrinsicOp::UnsignedSubSaturate, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    // --- Float intrinsics ---

    #[test]
    fn intrinsic_float_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F32, ConstValue::f32_from_native(3.5));
            let b = rb.constant(F32, ConstValue::f32_from_native(1.5));
            let res = rb.intrinsic(IntrinsicOp::FloatMin, state, &[a, b], F32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 1.5);
    }

    #[test]
    fn intrinsic_float_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F32, ConstValue::f32_from_native(3.5));
            let b = rb.constant(F32, ConstValue::f32_from_native(1.5));
            let res = rb.intrinsic(IntrinsicOp::FloatMax, state, &[a, b], F32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 3.5);
    }

    #[test]
    fn intrinsic_float_fma() {
        // fma(2.0, 3.0, 4.0) = 2*3 + 4 = 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F32, ConstValue::f32_from_native(2.0));
            let b = rb.constant(F32, ConstValue::f32_from_native(3.0));
            let c = rb.constant(F32, ConstValue::f32_from_native(4.0));
            let res = rb.intrinsic(IntrinsicOp::FloatFma, state, &[a, b, c], F32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 10.0);
    }

    #[test]
    fn intrinsic_float_copysign() {
        // copysign(5.0, -1.0) = -5.0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let mag = rb.constant(F32, ConstValue::f32_from_native(5.0));
            let sign = rb.constant(F32, ConstValue::f32_from_native(-1.0));
            let res = rb.intrinsic(IntrinsicOp::FloatCopySign, state, &[mag, sign], F32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), -5.0);
    }

    // --- Overflow-checked arithmetic ---

    #[test]
    fn intrinsic_sadd_overflow_no_overflow() {
        // 100 + 200 = 300, no overflow => return result
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(100);
            let b = rb.const_i32(200);
            let res = rb.intrinsic_overflow(IntrinsicOp::SignedAddOverflow, state, &[a, b], I32);
            FnResult {
                state: res.state,
                values: vec![res.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 300);
    }

    #[test]
    fn intrinsic_sadd_overflow_flag() {
        // i32::MAX + 1 overflows => overflow flag is true (1)
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(i32::MAX);
            let b = rb.const_i32(1);
            let res = rb.intrinsic_overflow(IntrinsicOp::SignedAddOverflow, state, &[a, b], I32);
            // overflow flag is i1, zero-extend to i32 to return it
            let flag = rb.cast(crate::rvsdg::CastOp::ZeroExtend, res.overflow, I32);
            FnResult {
                state: res.state,
                values: vec![flag],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn intrinsic_sadd_overflow_no_flag() {
        // 1 + 2 doesn't overflow => flag is 0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(1);
            let b = rb.const_i32(2);
            let res = rb.intrinsic_overflow(IntrinsicOp::SignedAddOverflow, state, &[a, b], I32);
            let flag = rb.cast(crate::rvsdg::CastOp::ZeroExtend, res.overflow, I32);
            FnResult {
                state: res.state,
                values: vec![flag],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    #[test]
    fn intrinsic_umul_overflow() {
        // Large unsigned multiply that overflows: 0x80000000 * 2 overflows u32
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(0x40000000);
            let b = rb.const_i32(4);
            let res = rb.intrinsic_overflow(IntrinsicOp::UnsignedMulOverflow, state, &[a, b], I32);
            let flag = rb.cast(crate::rvsdg::CastOp::ZeroExtend, res.overflow, I32);
            FnResult {
                state: res.state,
                values: vec![flag],
            }
        });
        // 0x40000000 * 4 = 0x100000000 which overflows u32
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }
}
