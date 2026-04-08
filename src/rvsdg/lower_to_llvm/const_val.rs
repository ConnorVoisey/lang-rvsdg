use crate::rvsdg::{
    ConstId, ConstValue, ConstantDef, ConstantKind, RVSDGMod,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
    types::TypeRef,
};
use inkwell::{
    AddressSpace,
    types::BasicTypeEnum,
    values::{BasicValue, BasicValueEnum},
};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_const_id<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        const_id: ConstId,
    ) -> BasicValueEnum<'ctx> {
        let constant = self.constants.get(const_id);
        self.lower_const_def(llvm_builder, mapper, constant)
    }

    #[inline]
    pub(crate) fn lower_const_def<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        constant: &ConstantDef,
    ) -> BasicValueEnum<'ctx> {
        match &constant.kind {
            ConstantKind::Scalar(const_value) => {
                self.lower_const_value(llvm_builder, mapper, const_value, constant.ty)
            }
            ConstantKind::Zero => self
                .type_to_basic_type_llvm(llvm_builder.context, constant.ty)
                .const_zero(),
            ConstantKind::Aggregate(const_ids_span) => {
                let element_ids = self.constants.get_aggregate_elements(*const_ids_span);
                let elements: Vec<BasicValueEnum<'ctx>> = element_ids
                    .iter()
                    .map(|&id| self.lower_const_id(llvm_builder, mapper, id))
                    .collect();
                match constant.ty {
                    TypeRef::Array(array_type_id) => {
                        let arr = self.types.get_array(array_type_id);
                        let elem_type =
                            self.type_to_basic_type_llvm(llvm_builder.context, arr.element);
                        BasicValueEnum::ArrayValue(match elem_type {
                            BasicTypeEnum::IntType(it) => {
                                let vals: Vec<_> =
                                    elements.iter().map(|e| e.into_int_value()).collect();
                                it.const_array(&vals)
                            }
                            BasicTypeEnum::FloatType(ft) => {
                                let vals: Vec<_> =
                                    elements.iter().map(|e| e.into_float_value()).collect();
                                ft.const_array(&vals)
                            }
                            BasicTypeEnum::PointerType(pt) => {
                                let vals: Vec<_> =
                                    elements.iter().map(|e| e.into_pointer_value()).collect();
                                pt.const_array(&vals)
                            }
                            BasicTypeEnum::StructType(st) => {
                                let vals: Vec<_> =
                                    elements.iter().map(|e| e.into_struct_value()).collect();
                                st.const_array(&vals)
                            }
                            BasicTypeEnum::ArrayType(at) => {
                                let vals: Vec<_> =
                                    elements.iter().map(|e| e.into_array_value()).collect();
                                at.const_array(&vals)
                            }
                            BasicTypeEnum::VectorType(vt) => {
                                let vals: Vec<_> =
                                    elements.iter().map(|e| e.into_vector_value()).collect();
                                vt.const_array(&vals)
                            }
                            BasicTypeEnum::ScalableVectorType(svt) => {
                                let vals: Vec<_> = elements
                                    .iter()
                                    .map(|e| e.into_scalable_vector_value())
                                    .collect();
                                svt.const_array(&vals)
                            }
                        })
                    }
                    _ => {
                        // Struct or other aggregate: use context.const_struct
                        BasicValueEnum::StructValue(
                            llvm_builder.context.const_struct(&elements, false),
                        )
                    }
                }
            }
            ConstantKind::String(items) => {
                BasicValueEnum::ArrayValue(llvm_builder.context.const_string(items, false))
            }
            ConstantKind::GlobalAddr(global_id) => mapper
                .get_global(*global_id)
                .expect("global should have been set during lower_globals")
                .as_basic_value_enum(),
            ConstantKind::Undef => todo!(),
        }
    }

    #[inline]
    pub(crate) fn lower_const_value<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        const_value: &ConstValue,
        ty: TypeRef,
    ) -> BasicValueEnum<'ctx> {
        match const_value {
            ConstValue::Int(val) => match ty {
                TypeRef::Scalar(scalar_type) => BasicValueEnum::IntValue(
                    scalar_type
                        .to_int_type(llvm_builder.context)
                        .const_int(*val as u64, false),
                ),
                t => unreachable!("int constant should have scalar type, got: {t:?}"),
            },
            ConstValue::F32(val) => BasicValueEnum::FloatValue(
                llvm_builder
                    .context
                    .f32_type()
                    .const_float(f32::from_bits(*val) as f64),
            ),
            ConstValue::F64(val) => BasicValueEnum::FloatValue(
                llvm_builder
                    .context
                    .f64_type()
                    .const_float(f64::from_bits(*val)),
            ),
            ConstValue::NullPtr => BasicValueEnum::PointerValue(
                llvm_builder
                    .context
                    .ptr_type(AddressSpace::default())
                    .const_null(),
            ),
            ConstValue::Poison => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, GlobalInit, GlobalLinkage, ICmpPred, RVSDGMod,
        func::{FnLinkageType, FnResult},
        lower_to_llvm::test_utils::test_utils::{
            jit_run_f32, jit_run_f64, jit_run_i32, jit_run_i64,
        },
        types::{ArrayType, BOOL, F32, F64, I8, I16, I32, I64, TypeRef},
        value::ConstValue,
    };

    #[test]
    fn const_i32_positive() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(42);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn const_i32_zero() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(0);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    #[test]
    fn const_i32_negative() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(-1);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), -1);
    }

    #[test]
    fn const_i32_max() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(i32::MAX);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), i32::MAX);
    }

    #[test]
    fn const_i32_min() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_i32(i32::MIN);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), i32::MIN);
    }

    #[test]
    fn const_bool_true() {
        // bool true (1) used as condition in ternary => returns 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(1));
            let t = rb.const_i32(10);
            let f = rb.const_i32(20);
            let v = rb.ternary(cond, t, f, I32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn const_bool_false() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let cond = rb.constant(BOOL, ConstValue::Int(0));
            let t = rb.const_i32(10);
            let f = rb.const_i32(20);
            let v = rb.ternary(cond, t, f, I32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 20);
    }

    #[test]
    fn const_i8_add() {
        // i8: 100 + 27 = 127, compare in i8
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(I8, ConstValue::Int(100));
            let b = rb.constant(I8, ConstValue::Int(27));
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I8);
            let expected = rb.constant(I8, ConstValue::Int(127));
            let cmp = rb.icmp(ICmpPred::Eq, sum, expected);
            let one = rb.const_i32(1);
            let zero = rb.const_i32(0);
            let v = rb.ternary(cmp, one, zero, I32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn const_i16_mul() {
        // i16: 200 * 3 = 600, compare in i16
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(I16, ConstValue::Int(200));
            let b = rb.constant(I16, ConstValue::Int(3));
            let product = rb.binary(BinaryOp::Mul, ArithFlags::default(), a, b, I16);
            let expected = rb.constant(I16, ConstValue::Int(600));
            let cmp = rb.icmp(ICmpPred::Eq, product, expected);
            let one = rb.const_i32(1);
            let zero = rb.const_i32(0);
            let v = rb.ternary(cmp, one, zero, I32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn const_i64_add() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I64], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i64(3_000_000_000);
            let b = rb.const_i64(1_000_000_000);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I64);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i64(&rvsdg, "test"), 4_000_000_000);
    }

    #[test]
    fn const_f32_value() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F32, ConstValue::f32_from_native(3.5));
            let b = rb.constant(F32, ConstValue::f32_from_native(1.5));
            let sum = rb.binary(BinaryOp::FloatAdd, ArithFlags::default(), a, b, F32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 5.0);
    }

    #[test]
    fn const_f32_negative() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F32, ConstValue::f32_from_native(-2.5));
            let b = rb.constant(F32, ConstValue::f32_from_native(2.5));
            let sum = rb.binary(BinaryOp::FloatAdd, ArithFlags::default(), a, b, F32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 0.0);
    }

    #[test]
    fn const_f64_value() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F64], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.constant(F64, ConstValue::f64_from_native(1.0));
            let b = rb.constant(F64, ConstValue::f64_from_native(2.0));
            let sum = rb.binary(BinaryOp::FloatAdd, ArithFlags::default(), a, b, F64);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_f64(&rvsdg, "test"), 3.0);
    }

    #[test]
    fn const_multiple_uses() {
        // Same constant used in multiple operations
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let five = rb.const_i32(5);
            // 5 + 5 = 10, then 10 * 5 = 50
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), five, five, I32);
            let result = rb.binary(BinaryOp::Mul, ArithFlags::default(), sum, five, I32);
            FnResult {
                state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 50);
    }

    #[test]
    fn const_string_extract_bytes() {
        // Create a string constant "Hi" and extract each byte
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I8,
            len: 2,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let str_id = rvsdg.constants.string(arr_ty, b"Hi".to_vec());
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let s = rb.const_pool_ref(str_id, arr_ty);
            let h = rb.extract_field(s, &[0], I8);
            let i = rb.extract_field(s, &[1], I8);
            // 'H' = 72, 'i' = 105, sum = 177
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), h, i, I8);
            // compare to expected
            let expected = rb.constant(I8, ConstValue::Int(177));
            let cmp = rb.icmp(ICmpPred::Eq, sum, expected);
            let one = rb.const_i32(1);
            let zero = rb.const_i32(0);
            let v = rb.ternary(cmp, one, zero, I32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn const_global_load() {
        // Define a global i32 constant initialized to 42, load it, return the value
        use crate::rvsdg::types::PtrType;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init_id = rvsdg.constants.scalar(I32, ConstValue::Int(42));
        let global_id = rvsdg.define_global(
            String::from("my_global"),
            I32,
            GlobalInit::Init(init_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty_id = rvsdg.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        });
        let ptr_ty = TypeRef::Ptr(ptr_ty_id);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    fn make_ptr_ty(rvsdg: &mut RVSDGMod, pointee: TypeRef) -> TypeRef {
        use crate::rvsdg::types::PtrType;
        let id = rvsdg.types.intern_ptr(PtrType {
            pointee: Some(pointee),
            alias_set: None,
            no_escape: false,
        });
        TypeRef::Ptr(id)
    }

    #[test]
    fn const_global_load_i64() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init_id = rvsdg.constants.scalar(I64, ConstValue::Int(9_000_000_000));
        let global_id = rvsdg.define_global(
            String::from("big_val"),
            I64,
            GlobalInit::Init(init_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I64);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I64], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, I64, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i64(&rvsdg, "test"), 9_000_000_000);
    }

    #[test]
    fn const_global_load_f32() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init_id = rvsdg
            .constants
            .scalar(F32, ConstValue::f32_from_native(3.14));
        let global_id = rvsdg.define_global(
            String::from("pi_approx"),
            F32,
            GlobalInit::Init(init_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, F32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, F32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 3.14f32);
    }

    #[test]
    fn const_global_load_negative() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init_id = rvsdg.constants.scalar(I32, ConstValue::Int(-999));
        let global_id = rvsdg.define_global(
            String::from("neg_val"),
            I32,
            GlobalInit::Init(init_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), -999);
    }

    #[test]
    fn const_global_load_add() {
        // Load two globals and add them
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init_a = rvsdg.constants.scalar(I32, ConstValue::Int(30));
        let init_b = rvsdg.constants.scalar(I32, ConstValue::Int(12));
        let global_a = rvsdg.define_global(
            String::from("a"),
            I32,
            GlobalInit::Init(init_a),
            true,
            GlobalLinkage::Internal,
        );
        let global_b = rvsdg.define_global(
            String::from("b"),
            I32,
            GlobalInit::Init(init_b),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr_a = rb.global_ref(global_a, ptr_ty);
            let la = rb.load(state, ptr_a, I32, None, false);
            let ptr_b = rb.global_ref(global_b, ptr_ty);
            let lb = rb.load(la.state, ptr_b, I32, None, false);
            let sum = rb.binary(
                BinaryOp::Add,
                ArithFlags::default(),
                la.value,
                lb.value,
                I32,
            );
            FnResult {
                state: lb.state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn const_global_load_in_loop() {
        // Load a global limit, loop up to it
        // limit = 7; x = 0; do { x++ } while (x < limit) => 7
        use crate::rvsdg::builder::LoopResult;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init_id = rvsdg.constants.scalar(I32, ConstValue::Int(7));
        let global_id = rvsdg.define_global(
            String::from("limit"),
            I32,
            GlobalInit::Init(init_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, I32, None, false);
            let limit = loaded.value;
            let x = rb.const_i32(0);
            let res = rb.theta(loaded.state, &[x], |rb| {
                let loop_x = rb.param(0);
                let one = rb.const_i32(1);
                let next_x = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_x, one, I32);
                let condition = rb.icmp(ICmpPred::SignedLt, next_x, limit);
                LoopResult {
                    condition,
                    next_state: state,
                    next_vars: vec![next_x],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 7);
    }

    #[test]
    fn const_aggregate_array_i32() {
        // Global [3 x i32] = [10, 20, 30], extract element at index 1 => 20
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 3,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let e0 = rvsdg.constants.scalar(I32, ConstValue::Int(10));
        let e1 = rvsdg.constants.scalar(I32, ConstValue::Int(20));
        let e2 = rvsdg.constants.scalar(I32, ConstValue::Int(30));
        let agg_id = rvsdg.constants.aggregate(arr_ty, &[e0, e1, e2]);
        let global_id = rvsdg.define_global(
            String::from("arr"),
            arr_ty,
            GlobalInit::Init(agg_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, arr_ty, None, false);
            let elem = rb.extract_field(loaded.value, &[1], I32);
            FnResult {
                state: loaded.state,
                values: vec![elem],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 20);
    }

    #[test]
    fn const_aggregate_array_sum() {
        // Global [4 x i32] = [1, 2, 3, 4], sum all elements => 10
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 4,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let elems: Vec<_> = (1..=4)
            .map(|v| rvsdg.constants.scalar(I32, ConstValue::Int(v)))
            .collect();
        let agg_id = rvsdg.constants.aggregate(arr_ty, &elems);
        let global_id = rvsdg.define_global(
            String::from("arr"),
            arr_ty,
            GlobalInit::Init(agg_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, arr_ty, None, false);
            let e0 = rb.extract_field(loaded.value, &[0], I32);
            let e1 = rb.extract_field(loaded.value, &[1], I32);
            let e2 = rb.extract_field(loaded.value, &[2], I32);
            let e3 = rb.extract_field(loaded.value, &[3], I32);
            let sum01 = rb.binary(BinaryOp::Add, ArithFlags::default(), e0, e1, I32);
            let sum23 = rb.binary(BinaryOp::Add, ArithFlags::default(), e2, e3, I32);
            let total = rb.binary(BinaryOp::Add, ArithFlags::default(), sum01, sum23, I32);
            FnResult {
                state: loaded.state,
                values: vec![total],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn const_aggregate_array_f32() {
        // Global [2 x f32] = [1.5, 2.5], extract and add => 4.0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: F32,
            len: 2,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let e0 = rvsdg
            .constants
            .scalar(F32, ConstValue::f32_from_native(1.5));
        let e1 = rvsdg
            .constants
            .scalar(F32, ConstValue::f32_from_native(2.5));
        let agg_id = rvsdg.constants.aggregate(arr_ty, &[e0, e1]);
        let global_id = rvsdg.define_global(
            String::from("arr"),
            arr_ty,
            GlobalInit::Init(agg_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, arr_ty, None, false);
            let a = rb.extract_field(loaded.value, &[0], F32);
            let b = rb.extract_field(loaded.value, &[1], F32);
            let sum = rb.binary(BinaryOp::FloatAdd, ArithFlags::default(), a, b, F32);
            FnResult {
                state: loaded.state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 4.0);
    }

    #[test]
    fn const_aggregate_struct() {
        // Anonymous struct { i32, i32 } = { 100, 200 }
        // Extract field 0 + field 1 => 300
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let e0 = rvsdg.constants.scalar(I32, ConstValue::Int(100));
        let e1 = rvsdg.constants.scalar(I32, ConstValue::Int(200));
        // Use State as a placeholder type since we don't have struct types wired up yet;
        // the lowering falls into the `_ =>` branch which uses context.const_struct
        let agg_id = rvsdg.constants.aggregate(TypeRef::State, &[e0, e1]);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let s = rb.const_pool_ref(agg_id, TypeRef::State);
            let a = rb.extract_field(s, &[0], I32);
            let b = rb.extract_field(s, &[1], I32);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 300);
    }

    #[test]
    fn const_zero_i32() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let zero_id = rvsdg.constants.zero(I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_pool_ref(zero_id, I32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    #[test]
    fn const_zero_f32() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let zero_id = rvsdg.constants.zero(F32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[F32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let v = rb.const_pool_ref(zero_id, F32);
            FnResult {
                state,
                values: vec![v],
            }
        });
        assert_eq!(jit_run_f32(&rvsdg, "test"), 0.0);
    }

    #[test]
    fn const_zero_array() {
        // Zero-initialized [3 x i32], all elements should be 0
        // Extract element 0 + element 2 => 0
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 3,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let zero_id = rvsdg.constants.zero(arr_ty);
        let global_id = rvsdg.define_global(
            String::from("zeroed"),
            arr_ty,
            GlobalInit::Init(zero_id),
            true,
            GlobalLinkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let loaded = rb.load(state, ptr, arr_ty, None, false);
            let e0 = rb.extract_field(loaded.value, &[0], I32);
            let e2 = rb.extract_field(loaded.value, &[2], I32);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), e0, e2, I32);
            FnResult {
                state: loaded.state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }

    #[test]
    fn const_zero_add_nonzero() {
        // zero(i32) + 42 => 42
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let zero_id = rvsdg.constants.zero(I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let z = rb.const_pool_ref(zero_id, I32);
            let v = rb.const_i32(42);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), z, v, I32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }
}
