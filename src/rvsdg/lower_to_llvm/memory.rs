use crate::rvsdg::{
    MemoryOrdering, RVSDGMod, ValueId,
    func::Function,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
    types::TypeRef,
};
use inkwell::{AtomicOrdering, builder::BuilderError, values::BasicValueEnum};

impl RVSDGMod {
    #[inline]
    pub(crate) fn lower_load<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        addr: ValueId,
        loaded_type: TypeRef,
        align: Option<u32>,
        volatile: bool,
        value_id: ValueId,
    ) -> Result<(), BuilderError> {
        let ptr = self.expect_value(llvm_builder, mapper, rvsdg_func, addr)?;
        let pointee_type = self.type_to_basic_type_llvm(llvm_builder.context, loaded_type);
        let load =
            llvm_builder
                .builder
                .build_load(pointee_type, ptr.into_pointer_value(), "load")?;
        // this is a gross way of getting the load value as an instruction but I don't know
        // of a better way.
        let inst = llvm_builder
            .builder
            .get_insert_block()
            .unwrap()
            .get_last_instruction()
            .unwrap();
        if let Some(a) = align {
            inst.set_alignment(a).unwrap();
        }
        if volatile {
            inst.set_volatile(true).unwrap();
        }
        // Load is a multi-output node (state + value). Write value to Project slot.
        let project_id = ValueId(value_id.0 + 1);
        mapper.set_val(project_id, load);

        Ok(())
    }

    #[inline]
    pub(crate) fn lower_store<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        addr: ValueId,
        value: ValueId,
        align: Option<u32>,
        volatile: bool,
    ) -> Result<(), BuilderError> {
        let ptr = self.expect_value(llvm_builder, mapper, rvsdg_func, addr)?;
        let llvm_val = self.expect_value(llvm_builder, mapper, rvsdg_func, value)?;
        let store = llvm_builder
            .builder
            .build_store(ptr.into_pointer_value(), llvm_val)?;

        if let Some(a) = align {
            store.set_alignment(a).unwrap();
        }
        if volatile {
            store.set_volatile(true).unwrap();
        }

        Ok(())
    }
    #[inline]
    pub(crate) fn lower_alloca<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        value_id: ValueId,
        elem_type: TypeRef,
        count: ValueId,
    ) -> Result<(), BuilderError> {
        let llvm_type = self.type_to_basic_type_llvm(llvm_builder.context, elem_type);
        let count_val = self.expect_value(llvm_builder, mapper, rvsdg_func, count)?;
        let alloca = llvm_builder.builder.build_array_alloca(
            llvm_type,
            count_val.into_int_value(),
            "alloca",
        )?;
        let ptr_val = BasicValueEnum::PointerValue(alloca);
        let project_id = ValueId(value_id.0 + 1);
        mapper.set_val(project_id, ptr_val);
        Ok(())
    }

    pub(crate) fn lower_cmpxchg<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        value_id: ValueId,
        addr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    ) -> Result<(), BuilderError> {
        let ptr = self.expect_value(llvm_builder, mapper, rvsdg_func, addr)?;
        let cmp = self.expect_value(llvm_builder, mapper, rvsdg_func, expected)?;
        let new = self.expect_value(llvm_builder, mapper, rvsdg_func, desired)?;

        let result = llvm_builder.builder.build_cmpxchg(
            ptr.into_pointer_value(),
            cmp,
            new,
            ordering_to_llvm(success_ordering),
            ordering_to_llvm(failure_ordering),
        )?;

        // cmpxchg returns {T, i1}: old value and success flag
        let old_val = llvm_builder
            .builder
            .build_extract_value(result, 0, "cas.old")?;
        let success = llvm_builder
            .builder
            .build_extract_value(result, 1, "cas.ok")?;

        let project_0 = ValueId(value_id.0 + 1);
        let project_1 = ValueId(value_id.0 + 2);
        mapper.set_val(project_0, old_val);
        mapper.set_val(project_1, success);
        Ok(())
    }
}

pub(crate) fn ordering_to_llvm(ordering: MemoryOrdering) -> AtomicOrdering {
    match ordering {
        MemoryOrdering::Relaxed => AtomicOrdering::Monotonic,
        MemoryOrdering::Acquire => AtomicOrdering::Acquire,
        MemoryOrdering::Release => AtomicOrdering::Release,
        MemoryOrdering::AcquireRelease => AtomicOrdering::AcquireRelease,
        MemoryOrdering::SequentiallyConsistent => AtomicOrdering::SequentiallyConsistent,
    }
}
#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, GlobalInit, ICmpPred, Linkage, RVSDGMod,
        builder::LoopResult,
        func::FnResult,
        lower_to_llvm::test_utils::test_utils::jit_run_i32,
        types::{I32, PtrType, TypeRef},
        value::ConstValue,
    };

    fn make_ptr_ty(rvsdg: &mut RVSDGMod, pointee: TypeRef) -> TypeRef {
        let id = rvsdg.types.intern_ptr(PtrType {
            pointee: Some(pointee),
            alias_set: None,
            no_escape: false,
        });
        TypeRef::Ptr(id)
    }

    // --- Store + Load via globals ---

    #[test]
    fn store_load_global() {
        // Mutable global initialized to 0, store 42, load it back
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let global_id = rvsdg.define_global(
            String::from("val"),
            I32,
            GlobalInit::Init(init),
            false,
            Linkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let val = rb.const_i32(42);
            let s1 = rb.store(state, ptr, val, None, false);
            let loaded = rb.load(s1, ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn store_overwrite_global() {
        // Store 10, then overwrite with 99, load should return 99
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let global_id = rvsdg.define_global(
            String::from("val"),
            I32,
            GlobalInit::Init(init),
            false,
            Linkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let ten = rb.const_i32(10);
            let s1 = rb.store(state, ptr, ten, None, false);
            let ninety_nine = rb.const_i32(99);
            let s2 = rb.store(s1, ptr, ninety_nine, None, false);
            let loaded = rb.load(s2, ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 99);
    }

    #[test]
    fn store_load_two_globals() {
        // Two globals, store different values, load and add them
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let g_a = rvsdg.define_global(
            String::from("a"),
            I32,
            GlobalInit::Init(init),
            false,
            Linkage::Internal,
        );
        let g_b = rvsdg.define_global(
            String::from("b"),
            I32,
            GlobalInit::Init(init),
            false,
            Linkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr_a = rb.global_ref(g_a, ptr_ty);
            let ptr_b = rb.global_ref(g_b, ptr_ty);
            let thirty = rb.const_i32(30);
            let twelve = rb.const_i32(12);
            let s1 = rb.store(state, ptr_a, thirty, None, false);
            let s2 = rb.store(s1, ptr_b, twelve, None, false);
            let la = rb.load(s2, ptr_a, I32, None, false);
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

    // --- Alloca + Store + Load ---

    #[test]
    fn alloca_store_load() {
        // Allocate stack i32, store 77, load it back
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let val = rb.const_i32(77);
            let s1 = rb.store(alloc.state, alloc.ptr, val, None, false);
            let loaded = rb.load(s1, alloc.ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 77);
    }

    #[test]
    fn alloca_accumulate_in_loop() {
        // Allocate i32 on stack, init to 0, loop 5 times adding 10 each time
        // Result: 50
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let zero = rb.const_i32(0);
            let s1 = rb.store(alloc.state, alloc.ptr, zero, None, false);

            // Loop counter as a loop var, accumulator via store/load on the alloca
            let i = rb.const_i32(0);
            let res = rb.theta(s1, &[i], |rb| {
                let loop_i = rb.param(0);
                // load current value, add 10, store back
                let cur = rb.load(s1, alloc.ptr, I32, None, false);
                let ten = rb.const_i32(10);
                let next_val = rb.binary(BinaryOp::Add, ArithFlags::default(), cur.value, ten, I32);
                let s2 = rb.store(cur.state, alloc.ptr, next_val, None, false);

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let five = rb.const_i32(5);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, five);
                LoopResult {
                    condition: cond,
                    next_state: s2,
                    next_vars: vec![next_i],
                }
            });

            let final_val = rb.load(res.state, alloc.ptr, I32, None, false);
            FnResult {
                state: final_val.state,
                values: vec![final_val.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 50);
    }

    #[test]
    fn alloca_swap() {
        // Allocate two stack i32s, store 10 and 20, swap via a temp, verify
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let a = rb.alloca(state, I32, one, ptr_ty);
            let b = rb.alloca(a.state, I32, one, ptr_ty);

            // a = 10, b = 20
            let ten = rb.const_i32(10);
            let twenty = rb.const_i32(20);
            let s1 = rb.store(b.state, a.ptr, ten, None, false);
            let s2 = rb.store(s1, b.ptr, twenty, None, false);

            // swap: tmp = *a; *a = *b; *b = tmp
            let tmp = rb.load(s2, a.ptr, I32, None, false);
            let b_val = rb.load(tmp.state, b.ptr, I32, None, false);
            let s3 = rb.store(b_val.state, a.ptr, b_val.value, None, false);
            let s4 = rb.store(s3, b.ptr, tmp.value, None, false);

            // *a should now be 20, *b should be 10
            // return *a - *b = 20 - 10 = 10
            let la = rb.load(s4, a.ptr, I32, None, false);
            let lb = rb.load(la.state, b.ptr, I32, None, false);
            let diff = rb.binary(
                BinaryOp::Sub,
                ArithFlags::default(),
                la.value,
                lb.value,
                I32,
            );
            FnResult {
                state: lb.state,
                values: vec![diff],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn store_global_in_loop() {
        // Mutable global, increment it in a loop
        // g = 0; for i in 0..10 { g = g + 3 } => g = 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let init = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let global_id = rvsdg.define_global(
            String::from("counter"),
            I32,
            GlobalInit::Init(init),
            false,
            Linkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let i = rb.const_i32(0);
            let res = rb.theta(state, &[i], |rb| {
                let loop_i = rb.param(0);
                let cur = rb.load(state, ptr, I32, None, false);
                let three = rb.const_i32(3);
                let next_val =
                    rb.binary(BinaryOp::Add, ArithFlags::default(), cur.value, three, I32);
                let s2 = rb.store(cur.state, ptr, next_val, None, false);

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let ten = rb.const_i32(10);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, ten);
                LoopResult {
                    condition: cond,
                    next_state: s2,
                    next_vars: vec![next_i],
                }
            });

            let final_val = rb.load(res.state, ptr, I32, None, false);
            FnResult {
                state: final_val.state,
                values: vec![final_val.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    // --- PtrOffset (GEP) ---

    #[test]
    fn gep_array_element_load() {
        // Global [4 x i32] = [10, 20, 30, 40], GEP to element 2, load => 30
        use crate::rvsdg::types::ArrayType;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 4,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let elems: Vec<_> = [10, 20, 30, 40]
            .iter()
            .map(|&v| rvsdg.constants.scalar(I32, ConstValue::Int(v)))
            .collect();
        let agg_id = rvsdg.constants.aggregate(arr_ty, &elems);
        let global_id = rvsdg.define_global(
            String::from("arr"),
            arr_ty,
            GlobalInit::Init(agg_id),
            true,
            Linkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let i32_ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let ptr = rb.global_ref(global_id, ptr_ty);
            let zero = rb.const_i32(0);
            let idx = rb.const_i32(2);
            // GEP [4 x i32]* ptr, 0, 2 => i32*
            let elem_ptr = rb.ptr_offset(ptr, arr_ty, &[zero, idx], i32_ptr_ty, true);
            let loaded = rb.load(state, elem_ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn gep_array_store_and_load() {
        // Alloca [3 x i32], store values via GEP, load them back and sum
        use crate::rvsdg::types::ArrayType;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 3,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let arr_ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let i32_ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, arr_ty, one, arr_ptr_ty);

            let zero = rb.const_i32(0);
            let idx0 = rb.const_i32(0);
            let idx1 = rb.const_i32(1);
            let idx2 = rb.const_i32(2);

            // GEP to each element and store
            let p0 = rb.ptr_offset(alloc.ptr, arr_ty, &[zero, idx0], i32_ptr_ty, true);
            let p1 = rb.ptr_offset(alloc.ptr, arr_ty, &[zero, idx1], i32_ptr_ty, true);
            let p2 = rb.ptr_offset(alloc.ptr, arr_ty, &[zero, idx2], i32_ptr_ty, true);

            let v100 = rb.const_i32(100);
            let v200 = rb.const_i32(200);
            let v300 = rb.const_i32(300);
            let s1 = rb.store(alloc.state, p0, v100, None, false);
            let s2 = rb.store(s1, p1, v200, None, false);
            let s3 = rb.store(s2, p2, v300, None, false);

            // Load all three and sum
            let l0 = rb.load(s3, p0, I32, None, false);
            let l1 = rb.load(l0.state, p1, I32, None, false);
            let l2 = rb.load(l1.state, p2, I32, None, false);
            let sum01 = rb.binary(
                BinaryOp::Add,
                ArithFlags::default(),
                l0.value,
                l1.value,
                I32,
            );
            let total = rb.binary(BinaryOp::Add, ArithFlags::default(), sum01, l2.value, I32);
            FnResult {
                state: l2.state,
                values: vec![total],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 600);
    }

    #[test]
    fn gep_dynamic_index_in_loop() {
        // Global [5 x i32] = [1, 2, 3, 4, 5], sum all elements via GEP in a loop => 15
        use crate::rvsdg::types::ArrayType;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 5,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let elems: Vec<_> = (1..=5)
            .map(|v| rvsdg.constants.scalar(I32, ConstValue::Int(v)))
            .collect();
        let agg_id = rvsdg.constants.aggregate(arr_ty, &elems);
        let global_id = rvsdg.define_global(
            String::from("arr"),
            arr_ty,
            GlobalInit::Init(agg_id),
            true,
            Linkage::Internal,
        );
        let ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let i32_ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let arr_ptr = rb.global_ref(global_id, ptr_ty);
            let zero_idx = rb.const_i32(0);
            // i = 0, sum = 0
            let i = rb.const_i32(0);
            let sum = rb.const_i32(0);
            let res = rb.theta(state, &[i, sum], |rb| {
                let loop_i = rb.param(0);
                let loop_sum = rb.param(1);

                // GEP arr[loop_i]
                let elem_ptr =
                    rb.ptr_offset(arr_ptr, arr_ty, &[zero_idx, loop_i], i32_ptr_ty, true);
                let loaded = rb.load(state, elem_ptr, I32, None, false);
                let next_sum = rb.binary(
                    BinaryOp::Add,
                    ArithFlags::default(),
                    loop_sum,
                    loaded.value,
                    I32,
                );

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let five = rb.const_i32(5);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, five);
                LoopResult {
                    condition: cond,
                    next_state: loaded.state,
                    next_vars: vec![next_i, next_sum],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(1)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 15);
    }

    // --- InsertField ---

    #[test]
    fn insert_field_array() {
        // Create [3 x i32] = [1, 2, 3], replace element 1 with 99, extract and sum
        // 1 + 99 + 3 = 103
        use crate::rvsdg::types::ArrayType;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 3,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let elems: Vec<_> = [1, 2, 3]
            .iter()
            .map(|&v| rvsdg.constants.scalar(I32, ConstValue::Int(v)))
            .collect();
        let agg_id = rvsdg.constants.aggregate(arr_ty, &elems);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let arr = rb.const_pool_ref(agg_id, arr_ty);
            let new_val = rb.const_i32(99);
            let modified = rb.insert_field(arr, new_val, &[1], arr_ty);
            let e0 = rb.extract_field(modified, &[0], I32);
            let e1 = rb.extract_field(modified, &[1], I32);
            let e2 = rb.extract_field(modified, &[2], I32);
            let sum01 = rb.binary(BinaryOp::Add, ArithFlags::default(), e0, e1, I32);
            let total = rb.binary(BinaryOp::Add, ArithFlags::default(), sum01, e2, I32);
            FnResult {
                state,
                values: vec![total],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 103);
    }

    #[test]
    fn insert_field_struct() {
        // Anonymous struct {i32, i32} = {10, 20}, replace field 0 with 50
        // Extract field 0 + field 1 = 50 + 20 = 70
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let e0 = rvsdg.constants.scalar(I32, ConstValue::Int(10));
        let e1 = rvsdg.constants.scalar(I32, ConstValue::Int(20));
        let agg_id = rvsdg.constants.aggregate(TypeRef::State, &[e0, e1]);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let s = rb.const_pool_ref(agg_id, TypeRef::State);
            let new_val = rb.const_i32(50);
            let modified = rb.insert_field(s, new_val, &[0], TypeRef::State);
            let a = rb.extract_field(modified, &[0], I32);
            let b = rb.extract_field(modified, &[1], I32);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 70);
    }

    #[test]
    fn insert_field_preserves_other_fields() {
        // {100, 200, 300}, insert 999 at index 2, verify index 0 and 1 unchanged
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let e0 = rvsdg.constants.scalar(I32, ConstValue::Int(100));
        let e1 = rvsdg.constants.scalar(I32, ConstValue::Int(200));
        let e2 = rvsdg.constants.scalar(I32, ConstValue::Int(300));
        let agg_id = rvsdg.constants.aggregate(TypeRef::State, &[e0, e1, e2]);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let s = rb.const_pool_ref(agg_id, TypeRef::State);
            let new_val = rb.const_i32(999);
            let modified = rb.insert_field(s, new_val, &[2], TypeRef::State);
            // field 0 should still be 100, field 1 should be 200
            let a = rb.extract_field(modified, &[0], I32);
            let b = rb.extract_field(modified, &[1], I32);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 300);
    }

    #[test]
    fn insert_field_chained() {
        // Start with {0, 0}, insert 10 at field 0, then insert 20 at field 1
        // Extract and sum => 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let e0 = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let e1 = rvsdg.constants.scalar(I32, ConstValue::Int(0));
        let agg_id = rvsdg.constants.aggregate(TypeRef::State, &[e0, e1]);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let s = rb.const_pool_ref(agg_id, TypeRef::State);
            let ten = rb.const_i32(10);
            let twenty = rb.const_i32(20);
            let s1 = rb.insert_field(s, ten, &[0], TypeRef::State);
            let s2 = rb.insert_field(s1, twenty, &[1], TypeRef::State);
            let a = rb.extract_field(s2, &[0], I32);
            let b = rb.extract_field(s2, &[1], I32);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    // --- CompareAndSwap ---

    #[test]
    fn cmpxchg_success() {
        // alloca i32, store 10, CAS(expected=10, desired=42) => succeeds, old=10
        // return old value
        use crate::rvsdg::MemoryOrdering;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let ten = rb.const_i32(10);
            let s1 = rb.store(alloc.state, alloc.ptr, ten, None, false);

            let expected = rb.const_i32(10);
            let desired = rb.const_i32(42);
            let cas = rb.compare_and_swap(
                s1,
                alloc.ptr,
                expected,
                desired,
                MemoryOrdering::SequentiallyConsistent,
                MemoryOrdering::SequentiallyConsistent,
                I32,
            );
            // old_value should be 10 (what was there before)
            FnResult {
                state: cas.state,
                values: vec![cas.old_value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn cmpxchg_success_new_value() {
        // CAS succeeds, then load to verify the new value was stored
        use crate::rvsdg::MemoryOrdering;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let ten = rb.const_i32(10);
            let s1 = rb.store(alloc.state, alloc.ptr, ten, None, false);

            let expected = rb.const_i32(10);
            let desired = rb.const_i32(42);
            let cas = rb.compare_and_swap(
                s1,
                alloc.ptr,
                expected,
                desired,
                MemoryOrdering::SequentiallyConsistent,
                MemoryOrdering::SequentiallyConsistent,
                I32,
            );
            // load should now return 42
            let loaded = rb.load(cas.state, alloc.ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn cmpxchg_failure() {
        // alloca i32, store 10, CAS(expected=99, desired=42) => fails
        // value should remain 10
        use crate::rvsdg::MemoryOrdering;
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let ten = rb.const_i32(10);
            let s1 = rb.store(alloc.state, alloc.ptr, ten, None, false);

            let wrong_expected = rb.const_i32(99);
            let desired = rb.const_i32(42);
            let cas = rb.compare_and_swap(
                s1,
                alloc.ptr,
                wrong_expected,
                desired,
                MemoryOrdering::SequentiallyConsistent,
                MemoryOrdering::SequentiallyConsistent,
                I32,
            );
            // CAS failed, value unchanged, load should return 10
            let loaded = rb.load(cas.state, alloc.ptr, I32, None, false);
            FnResult {
                state: loaded.state,
                values: vec![loaded.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 10);
    }

    #[test]
    fn cmpxchg_success_flag() {
        // CAS(expected=10, actual=10) => success flag = true (1)
        use crate::rvsdg::{CastOp, MemoryOrdering};
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let ten = rb.const_i32(10);
            let s1 = rb.store(alloc.state, alloc.ptr, ten, None, false);

            let expected = rb.const_i32(10);
            let desired = rb.const_i32(42);
            let cas = rb.compare_and_swap(
                s1,
                alloc.ptr,
                expected,
                desired,
                MemoryOrdering::SequentiallyConsistent,
                MemoryOrdering::SequentiallyConsistent,
                I32,
            );
            let flag = rb.cast(CastOp::ZeroExtend, cas.success, I32);
            FnResult {
                state: cas.state,
                values: vec![flag],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }

    #[test]
    fn cmpxchg_failure_flag() {
        // CAS(expected=99, actual=10) => success flag = false (0)
        use crate::rvsdg::{CastOp, MemoryOrdering};
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], Linkage::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, I32, one, ptr_ty);
            let ten = rb.const_i32(10);
            let s1 = rb.store(alloc.state, alloc.ptr, ten, None, false);

            let wrong = rb.const_i32(99);
            let desired = rb.const_i32(42);
            let cas = rb.compare_and_swap(
                s1,
                alloc.ptr,
                wrong,
                desired,
                MemoryOrdering::SequentiallyConsistent,
                MemoryOrdering::SequentiallyConsistent,
                I32,
            );
            let flag = rb.cast(CastOp::ZeroExtend, cas.success, I32);
            FnResult {
                state: cas.state,
                values: vec![flag],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 0);
    }
}
