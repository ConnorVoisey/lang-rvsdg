use crate::rvsdg::{
    FuncId, GlobalId, GlobalInit, GlobalLinkage, RVSDGMod, Region, ValueId, ValueKind,
    func::{FnLinkageType, Function},
    types::{ScalarType, TypeRef},
};
use inkwell::{
    AddressSpace, OptimizationLevel,
    builder::{Builder, BuilderError},
    context::Context,
    module::{Linkage, Module},
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType, IntType},
    values::{BasicValue, BasicValueEnum, FunctionValue, GlobalValue},
};
use std::{error::Error, path::Path, process::Command};

pub mod binary;
pub mod cast;
pub mod const_val;
pub mod gamma;
pub mod intrinsic;
pub mod memory;
pub mod test_utils;
pub mod theta;
pub mod unary;
pub mod value;

#[derive(Debug)]
pub struct LLVMBuilderCtx<'a, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    builder: &'a Builder<'ctx>,
}

#[derive(Debug)]
pub struct ValueMapper<'ctx> {
    values: Vec<Option<BasicValueEnum<'ctx>>>,
    fns: Vec<Option<FunctionValue<'ctx>>>,
    globals: Vec<Option<GlobalValue<'ctx>>>,
}

impl<'ctx> ValueMapper<'ctx> {
    fn new(rvsdg_mod: &RVSDGMod) -> Self {
        Self {
            values: vec![None; rvsdg_mod.values.len()],
            fns: vec![None; rvsdg_mod.functions.len()],
            globals: vec![None; rvsdg_mod.globals.len()],
        }
    }

    fn get_val(&self, value_id: ValueId) -> &Option<BasicValueEnum<'ctx>> {
        &self.values[value_id.0 as usize]
    }
    fn set_val(&mut self, value_id: ValueId, value_enum: BasicValueEnum<'ctx>) {
        self.values[value_id.0 as usize] = Some(value_enum);
    }

    fn get_fn(&self, func_id: FuncId) -> &Option<FunctionValue<'ctx>> {
        &self.fns[func_id.0 as usize]
    }
    fn set_fn(&mut self, func_id: FuncId, func: FunctionValue<'ctx>) {
        self.fns[func_id.0 as usize] = Some(func);
    }

    fn get_global(&self, global_id: GlobalId) -> &Option<GlobalValue<'ctx>> {
        &self.globals[global_id.0 as usize]
    }
    fn set_global(&mut self, global_id: GlobalId, global_value: GlobalValue<'ctx>) {
        self.globals[global_id.0 as usize] = Some(global_value);
    }
}

impl RVSDGMod {
    /// Lower the RVSDG module into an LLVM module without emitting files.
    /// The caller owns the context and module lifetime.
    pub fn lower_to_llvm_module<'ctx>(
        &self,
        context: &'ctx Context,
    ) -> Result<Module<'ctx>, BuilderError> {
        let module = context.create_module(&self.mod_name);
        let builder = context.create_builder();
        let llvm_builder = LLVMBuilderCtx {
            context,
            module: &module,
            builder: &builder,
        };
        let mut value_mapper = ValueMapper::new(self);
        self.lower_mod(&llvm_builder, &mut value_mapper)?;
        Ok(module)
    }

    pub fn output_with_llvm(&self) -> Result<(), Box<dyn Error>> {
        // initialise things
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");

        let context = Context::create();
        let module = context.create_module(&self.mod_name);
        let builder = context.create_builder();

        // actually convert the RVSDG module into LLVM
        let llvm_builder = LLVMBuilderCtx {
            context: &context,
            module: &module,
            builder: &builder,
        };
        let mut value_mapper = ValueMapper::new(self);
        self.lower_mod(&llvm_builder, &mut value_mapper)?;

        // more output things
        let llvm_triple = TargetTriple::create(&self.target.to_string());
        let target = Target::from_triple(&llvm_triple).expect("Failed to get target from triple");

        let machine = target
            .create_target_machine(
                &llvm_triple,
                "generic",
                "",
                OptimizationLevel::Default,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .expect("Failed to create target machine");

        let obj_file = format!("{}.o", self.mod_name);
        let obj_path = Path::new(&obj_file);
        machine
            .write_to_file(&module, FileType::Object, obj_path)
            .expect("Failed to write object file");

        println!("Wrote object file: {}", obj_path.display());

        // link, this will need to be conditional depending on the mode
        let exe_path = &self.mod_name;
        let status = Command::new("cc")
            .args([obj_path.to_str().unwrap(), "-o", exe_path])
            .status()
            .expect("Failed to invoke linker (cc)");

        if status.success() {
            println!("Linked executable: ./{exe_path}");
            println!("\nRun it with:  ./{exe_path}");
        } else {
            eprintln!("Linking failed with status: {status}");
        }
        Ok(())
    }

    fn lower_mod<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
    ) -> Result<(), BuilderError> {
        // For now we'll use a naive implementation that converts the RVSDG directly to llvm
        // without using predicates.
        // TODO: replace this implemenation with this https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43246.pdf

        self.lower_globals(llvm_builder, mapper);
        for func in self.functions.iter() {
            self.register_fn(llvm_builder, mapper, func);
        }
        for func in self.functions.iter() {
            if func.lambda_val.is_none() {
                continue; // declaration only, no body to lower
            }
            self.lower_fn(llvm_builder, mapper, func)?;
        }
        llvm_builder
            .module
            .verify()
            .expect("Module verification failed");

        Ok(())
    }

    fn lower_globals<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
    ) {
        for (i, global) in self.globals.iter().enumerate() {
            let llvm_type = self.type_to_basic_type_llvm(llvm_builder.context, global.ty);
            let glob = llvm_builder
                .module
                // TODO: replace the address space with the RVSDG address space,
                // Inkwell stores this as an i16, we store as a string
                .add_global(llvm_type, None, &global.name);
            glob.set_constant(global.is_constant);
            glob.set_linkage(global.linkage.to_llvm());
            match global.initializer {
                GlobalInit::Extern => (),
                GlobalInit::Init(const_id) => {
                    let const_val = self.lower_const_id(llvm_builder, mapper, const_id);
                    glob.set_initializer(&const_val as &dyn BasicValue);
                }
            };
            mapper.set_global(GlobalId(i as u32), glob);
        }
    }

    fn register_fn<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
    ) {
        assert!(
            rvsdg_func.return_types.len() < 2,
            "LLVM does not support more than one return value"
        );

        let param_types = rvsdg_func
            .params
            .iter()
            .map(|param| self.type_to_basic_meta_llvm(llvm_builder.context, param.ty))
            .collect::<Vec<_>>();
        let llvm_fn_type = if let Some(&ret_ty) = rvsdg_func.return_types.first() {
            self.type_to_basic_type_llvm(llvm_builder.context, ret_ty)
                .fn_type(&param_types, rvsdg_func.is_var_arg)
        } else {
            llvm_builder
                .context
                .void_type()
                .fn_type(&param_types, rvsdg_func.is_var_arg)
        };

        let func_ty = llvm_builder.module.add_function(
            &rvsdg_func.name,
            llvm_fn_type,
            Some(rvsdg_func.linkage_type.to_llvm()),
        );
        mapper.set_fn(rvsdg_func.id, func_ty);
    }

    fn lower_fn<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
    ) -> Result<(), BuilderError> {
        let func = mapper.get_fn(rvsdg_func.id).expect("register_fn should have been called before lower_fn which should have registered the function");
        let entry = llvm_builder.context.append_basic_block(func, "entry");
        llvm_builder.builder.position_at_end(entry);
        let fn_val = rvsdg_func
            .lambda_val
            .expect("lambda_val should have been set during RVSDG construction");
        let lambda_val = &self.values[fn_val.0 as usize];
        match &lambda_val.kind {
            ValueKind::Lambda {
                region: region_id,
                func_id: _,
            } => {
                // register the regions inputs to the llvm functions parameters so that they can be
                // referenced by project inside the region
                let region = &self.regions[region_id.0 as usize];
                for i in 0..region.params.len as u32 {
                    let param_id = ValueId(region.params.start + i);
                    mapper.set_val(param_id, func.get_nth_param(i).unwrap());
                }

                self.lower_region(llvm_builder, mapper, rvsdg_func, region)?;

                // regions results should be added from inside lower_region
                let res = self.value_pool.get(region.results);
                match res.len() {
                    0 => llvm_builder.builder.build_return(None)?,
                    1 => {
                        let val = mapper.get_val(res[0]).unwrap();
                        llvm_builder
                            .builder
                            .build_return(Some(&val as &dyn BasicValue))?
                    }
                    _ => panic!("LLVM does not support more than one return value"),
                }
            }
            t => unreachable!("lambda value has a value kind of {t:?}"),
        };
        Ok(())
    }

    fn lower_region<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        region: &Region,
    ) -> Result<(), BuilderError> {
        for &value_id in region.nodes.iter() {
            self.lower_value(llvm_builder, mapper, rvsdg_func, value_id)?;
        }
        Ok(())
    }
    fn type_to_basic_type_llvm<'a, 'b>(
        &self,
        context: &'b Context,
        ty: TypeRef,
    ) -> BasicTypeEnum<'b> {
        match ty {
            TypeRef::State => unreachable!(),
            TypeRef::Scalar(scalar_type) => match scalar_type {
                ScalarType::Bool => BasicTypeEnum::IntType(context.bool_type()),
                ScalarType::I8 => BasicTypeEnum::IntType(context.i8_type()),
                ScalarType::I16 => BasicTypeEnum::IntType(context.i16_type()),
                ScalarType::I32 => BasicTypeEnum::IntType(context.i32_type()),
                ScalarType::I64 => BasicTypeEnum::IntType(context.i64_type()),
                ScalarType::I128 => BasicTypeEnum::IntType(context.i128_type()),
                ScalarType::F32 => BasicTypeEnum::FloatType(context.f32_type()),
                ScalarType::F64 => BasicTypeEnum::FloatType(context.f64_type()),
                // Void is not a BasicType in LLVM — it only appears as a function
                // return type, never as a value/parameter/alloca type.
                ScalarType::Void => unreachable!("void is not a basic type"),
            },
            TypeRef::Ptr(_) => {
                BasicTypeEnum::PointerType(context.ptr_type(AddressSpace::default()))
            }
            TypeRef::Array(array_type_id) => {
                let arr = self.types.get_array(array_type_id);
                let elem = self.type_to_basic_type_llvm(context, arr.element);
                BasicTypeEnum::ArrayType(elem.array_type(arr.len as u32))
            }
            TypeRef::Struct(struct_id) => {
                let def = self.types.get_struct(struct_id);
                let field_types: Vec<BasicTypeEnum> = def
                    .fields
                    .iter()
                    .map(|f| self.type_to_basic_type_llvm(context, f.field_type))
                    .collect();
                BasicTypeEnum::StructType(context.struct_type(&field_types, false))
            }
            TypeRef::Vector(vector_type_id) => {
                let vec = self.types.get_vector(vector_type_id);
                let elem = self.type_to_basic_type_llvm(context, vec.element);
                match elem {
                    BasicTypeEnum::IntType(t) => BasicTypeEnum::VectorType(t.vec_type(vec.lanes)),
                    BasicTypeEnum::FloatType(t) => BasicTypeEnum::VectorType(t.vec_type(vec.lanes)),
                    BasicTypeEnum::PointerType(t) => {
                        BasicTypeEnum::VectorType(t.vec_type(vec.lanes))
                    }
                    _ => panic!("vector element must be scalar or pointer"),
                }
            }
            // FuncType is not a BasicType — functions exist only as pointers
            // (opaque ptr in LLVM 17+). If a TypeRef::Func reaches here, the
            // caller has a bug.
            TypeRef::Func(_) => unreachable!("function type is not a basic type"),
        }
    }

    fn type_to_basic_meta_llvm<'a, 'b>(
        &self,
        context: &'b Context,
        ty: TypeRef,
    ) -> BasicMetadataTypeEnum<'b> {
        match ty {
            // State is an IR-only concept with no LLVM representation.
            TypeRef::State => unreachable!("state has no LLVM type representation"),
            TypeRef::Scalar(scalar_type) => match scalar_type {
                ScalarType::Bool => BasicMetadataTypeEnum::IntType(context.bool_type()),
                ScalarType::I8 => BasicMetadataTypeEnum::IntType(context.i8_type()),
                ScalarType::I16 => BasicMetadataTypeEnum::IntType(context.i16_type()),
                ScalarType::I32 => BasicMetadataTypeEnum::IntType(context.i32_type()),
                ScalarType::I64 => BasicMetadataTypeEnum::IntType(context.i64_type()),
                ScalarType::I128 => BasicMetadataTypeEnum::IntType(context.i128_type()),
                ScalarType::F32 => BasicMetadataTypeEnum::FloatType(context.f32_type()),
                ScalarType::F64 => BasicMetadataTypeEnum::FloatType(context.f64_type()),
                ScalarType::Void => unreachable!("void is not a basic metadata type"),
            },
            TypeRef::Ptr(_) => {
                BasicMetadataTypeEnum::PointerType(context.ptr_type(AddressSpace::default()))
            }
            TypeRef::Array(array_type_id) => {
                let arr = self.types.get_array(array_type_id);
                let elem = self.type_to_basic_type_llvm(context, arr.element);
                BasicMetadataTypeEnum::ArrayType(elem.array_type(arr.len as u32))
            }
            TypeRef::Struct(struct_id) => {
                let def = self.types.get_struct(struct_id);
                let field_types: Vec<BasicTypeEnum> = def
                    .fields
                    .iter()
                    .map(|f| self.type_to_basic_type_llvm(context, f.field_type))
                    .collect();
                BasicMetadataTypeEnum::StructType(context.struct_type(&field_types, false))
            }
            TypeRef::Vector(vector_type_id) => {
                let vec = self.types.get_vector(vector_type_id);
                let elem = self.type_to_basic_type_llvm(context, vec.element);
                match elem {
                    BasicTypeEnum::IntType(t) => {
                        BasicMetadataTypeEnum::VectorType(t.vec_type(vec.lanes))
                    }
                    BasicTypeEnum::FloatType(t) => {
                        BasicMetadataTypeEnum::VectorType(t.vec_type(vec.lanes))
                    }
                    BasicTypeEnum::PointerType(t) => {
                        BasicMetadataTypeEnum::VectorType(t.vec_type(vec.lanes))
                    }
                    _ => panic!("vector element must be scalar or pointer"),
                }
            }
            TypeRef::Func(_) => unreachable!("function type is not a basic metadata type"),
        }
    }
}

impl FnLinkageType {
    fn to_llvm(&self) -> Linkage {
        match self {
            FnLinkageType::Internal => Linkage::Internal,
            FnLinkageType::External => Linkage::External,
            FnLinkageType::LinkOnce => Linkage::LinkOnceAny,
            FnLinkageType::LinkOnceODR => Linkage::LinkOnceODR,
            FnLinkageType::Weak => Linkage::WeakAny,
            FnLinkageType::WeakODR => Linkage::WeakODR,
            FnLinkageType::AvailableExternally => Linkage::AvailableExternally,
        }
    }
}

impl GlobalLinkage {
    fn to_llvm(&self) -> Linkage {
        match self {
            GlobalLinkage::Internal => Linkage::Internal,
            GlobalLinkage::External => Linkage::External,
            GlobalLinkage::LinkOnce => Linkage::LinkOnceAny,
            GlobalLinkage::Weak => Linkage::WeakAny,
            GlobalLinkage::Common => Linkage::Common,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, CastOp, GlobalInit, GlobalLinkage, ICmpPred, IntrinsicOp, RVSDGMod,
        builder::{BranchResult, LoopResult},
        func::{FnLinkageType, FnResult},
        lower_to_llvm::test_utils::test_utils::jit_run_i32,
        types::{ArrayType, I32, I64, PtrType, StructDef, StructField, TypeRef},
        value::ConstValue,
    };

    fn make_ptr_ty(rvsdg: &mut RVSDGMod, pointee: TypeRef) -> TypeRef {
        TypeRef::Ptr(rvsdg.types.intern_ptr(PtrType {
            pointee: Some(pointee),
            alias_set: None,
            no_escape: false,
        }))
    }

    #[test]
    fn struct_alloca_gep_store_load() {
        // Allocate a struct { i32, i32 } on the stack, store fields via GEP, load and sum
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let struct_id = rvsdg.types.intern_struct(StructDef {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    offset: 0,
                    field_type: I32,
                },
                StructField {
                    name: None,
                    offset: 4,
                    field_type: I32,
                },
            ],
            size: 8,
        });
        let struct_ty = TypeRef::Struct(struct_id);
        let struct_ptr_ty = make_ptr_ty(&mut rvsdg, struct_ty);
        let i32_ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let one = rb.const_i32(1);
            let alloc = rb.alloca(state, struct_ty, one, struct_ptr_ty);

            let zero = rb.const_i32(0);
            let idx1 = rb.const_i32(1);

            // GEP to field 0 and field 1
            let f0_ptr = rb.ptr_offset(alloc.ptr, struct_ty, &[zero, zero], i32_ptr_ty, true);
            let f1_ptr = rb.ptr_offset(alloc.ptr, struct_ty, &[zero, idx1], i32_ptr_ty, true);

            let v30 = rb.const_i32(30);
            let v12 = rb.const_i32(12);
            let s1 = rb.store(alloc.state, f0_ptr, v30, None, false);
            let s2 = rb.store(s1, f1_ptr, v12, None, false);

            let l0 = rb.load(s2, f0_ptr, I32, None, false);
            let l1 = rb.load(l0.state, f1_ptr, I32, None, false);
            let sum = rb.binary(
                BinaryOp::Add,
                ArithFlags::default(),
                l0.value,
                l1.value,
                I32,
            );
            FnResult {
                state: l1.state,
                values: vec![sum],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn multi_function_call() {
        // Define add(a, b) -> i32, call it from main
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let add_fn = rvsdg.declare_fn(
            String::from("add"),
            &[I32, I32],
            &[I32],
            FnLinkageType::Internal,
        );
        rvsdg.define_fn(add_fn, |rb, state| {
            let a = rb.param(0);
            let b = rb.param(1);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state,
                values: vec![sum],
            }
        });

        let main_fn = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(main_fn, |rb, state| {
            let a = rb.const_i32(30);
            let b = rb.const_i32(12);
            let res = rb.call(add_fn, state, &[a, b]);
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 42);
    }

    #[test]
    fn loop_with_conditional_accumulator() {
        // sum = 0; for i in 1..=10 { if i % 2 == 0 { sum += i } else { sum += 1 } }
        // even: 2+4+6+8+10 = 30, odd count: 5
        // total = 30 + 5 = 35
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let i = rb.const_i32(1);
            let sum = rb.const_i32(0);
            let res = rb.theta(state, &[i, sum], |rb| {
                let loop_i = rb.param(0);
                let loop_sum = rb.param(1);

                let two = rb.const_i32(2);
                let rem = rb.binary(
                    BinaryOp::UnsignedRem,
                    ArithFlags::default(),
                    loop_i,
                    two,
                    I32,
                );
                let zero = rb.const_i32(0);
                let is_even = rb.icmp(ICmpPred::Eq, rem, zero);

                // if even { i } else { 1 }
                let branch = rb.gamma(
                    is_even,
                    state,
                    &[loop_i],
                    |rb| BranchResult {
                        state,
                        values: vec![rb.param(0)],
                    },
                    |rb| BranchResult {
                        state,
                        values: vec![rb.const_i32(1)],
                    },
                );
                let to_add = branch.result(0);
                let next_sum =
                    rb.binary(BinaryOp::Add, ArithFlags::default(), loop_sum, to_add, I32);

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let ten = rb.const_i32(10);
                let cond = rb.icmp(ICmpPred::SignedLe, next_i, ten);
                LoopResult {
                    condition: cond,
                    next_state: state,
                    next_vars: vec![next_i, next_sum],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(1)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 35);
    }

    #[test]
    fn array_of_structs_via_globals() {
        // Global array of 3 structs {i32, i32}, sum field 1 of each
        // [{10, 1}, {20, 2}, {30, 3}] => sum of field 1 = 1 + 2 + 3 = 6
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let struct_id = rvsdg.types.intern_struct(StructDef {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    offset: 0,
                    field_type: I32,
                },
                StructField {
                    name: None,
                    offset: 4,
                    field_type: I32,
                },
            ],
            size: 8,
        });
        let struct_ty = TypeRef::Struct(struct_id);
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: struct_ty,
            len: 3,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);

        // Build aggregate: [{10,1}, {20,2}, {30,3}]
        let s0_f0 = rvsdg.constants.scalar(I32, ConstValue::Int(10));
        let s0_f1 = rvsdg.constants.scalar(I32, ConstValue::Int(1));
        let s0 = rvsdg.constants.aggregate(struct_ty, &[s0_f0, s0_f1]);
        let s1_f0 = rvsdg.constants.scalar(I32, ConstValue::Int(20));
        let s1_f1 = rvsdg.constants.scalar(I32, ConstValue::Int(2));
        let s1 = rvsdg.constants.aggregate(struct_ty, &[s1_f0, s1_f1]);
        let s2_f0 = rvsdg.constants.scalar(I32, ConstValue::Int(30));
        let s2_f1 = rvsdg.constants.scalar(I32, ConstValue::Int(3));
        let s2 = rvsdg.constants.aggregate(struct_ty, &[s2_f0, s2_f1]);
        let arr_const = rvsdg.constants.aggregate(arr_ty, &[s0, s1, s2]);

        let global_id = rvsdg.define_global(
            String::from("structs"),
            arr_ty,
            GlobalInit::Init(arr_const),
            true,
            GlobalLinkage::Internal,
        );
        let arr_ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let i32_ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let arr_ptr = rb.global_ref(global_id, arr_ptr_ty);
            let zero = rb.const_i32(0);
            let field1_idx = rb.const_i32(1);

            // Sum field 1 of each struct in a loop
            let i = rb.const_i32(0);
            let sum = rb.const_i32(0);
            let res = rb.theta(state, &[i, sum], |rb| {
                let loop_i = rb.param(0);
                let loop_sum = rb.param(1);

                // GEP arr[i].field1
                let field_ptr = rb.ptr_offset(
                    arr_ptr,
                    arr_ty,
                    &[zero, loop_i, field1_idx],
                    i32_ptr_ty,
                    true,
                );
                let loaded = rb.load(state, field_ptr, I32, None, false);
                let next_sum = rb.binary(
                    BinaryOp::Add,
                    ArithFlags::default(),
                    loop_sum,
                    loaded.value,
                    I32,
                );

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let three = rb.const_i32(3);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, three);
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
        assert_eq!(jit_run_i32(&rvsdg, "test"), 6);
    }

    #[test]
    fn cast_loop_with_i64_accumulator() {
        // Sum 1..10 using i64 accumulator, truncate back to i32
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let i = rb.const_i32(1);
            let sum = rb.const_i64(0);
            let res = rb.theta(state, &[i, sum], |rb| {
                let loop_i = rb.param(0);
                let loop_sum = rb.param(1);

                // widen i to i64, add to sum
                let wide_i = rb.cast(CastOp::SignExtend, loop_i, I64);
                let next_sum =
                    rb.binary(BinaryOp::Add, ArithFlags::default(), loop_sum, wide_i, I64);

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let ten = rb.const_i32(10);
                let cond = rb.icmp(ICmpPred::SignedLe, next_i, ten);
                LoopResult {
                    condition: cond,
                    next_state: state,
                    next_vars: vec![next_i, next_sum],
                }
            });
            // truncate i64 sum back to i32
            let result = rb.cast(CastOp::Truncate, res.result(1), I32);
            FnResult {
                state: res.state,
                values: vec![result],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 55);
    }

    #[test]
    fn fibonacci() {
        // fib(10) = 55 via theta loop
        // a = 0, b = 1; for _ in 0..10 { tmp = a + b; a = b; b = tmp }; return a
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let a = rb.const_i32(0);
            let b = rb.const_i32(1);
            let i = rb.const_i32(0);
            let res = rb.theta(state, &[a, b, i], |rb| {
                let loop_a = rb.param(0);
                let loop_b = rb.param(1);
                let loop_i = rb.param(2);

                let next_a = loop_b;
                let next_b = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_a, loop_b, I32);

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let ten = rb.const_i32(10);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, ten);
                LoopResult {
                    condition: cond,
                    next_state: state,
                    next_vars: vec![next_a, next_b, next_i],
                }
            });
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 55);
    }

    #[test]
    fn saturating_clamp_in_loop() {
        // Repeatedly add 100 using saturating add, 30 times
        // Without saturation: 3000. With i32 saturation: still 3000 (no overflow)
        // Then do it with a value close to MAX to actually saturate
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let val = rb.const_i32(i32::MAX - 50);
            let i = rb.const_i32(0);
            let res = rb.theta(state, &[val, i], |rb| {
                let loop_val = rb.param(0);
                let loop_i = rb.param(1);

                let hundred = rb.const_i32(100);
                let sat_res = rb.intrinsic(
                    IntrinsicOp::SignedAddSaturate,
                    state,
                    &[loop_val, hundred],
                    I32,
                );

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), loop_i, one, I32);
                let three = rb.const_i32(3);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, three);
                LoopResult {
                    condition: cond,
                    next_state: sat_res.state,
                    next_vars: vec![sat_res.value, next_i],
                }
            });
            // After 3 iterations of adding 100 starting from MAX-50,
            // first add: MAX-50+100 = MAX+50 -> saturates to MAX
            // subsequent adds also saturate to MAX
            FnResult {
                state: res.state,
                values: vec![res.result(0)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), i32::MAX);
    }

    #[test]
    fn nested_gamma_with_arithmetic() {
        // Nested conditionals:
        // x = 15
        // if x > 10 {
        //     if x > 20 { 100 } else { x * 2 }
        // } else {
        //     0
        // }
        // => 15 > 10 is true, 15 > 20 is false, so x * 2 = 30
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let x = rb.const_i32(15);
            let ten = rb.const_i32(10);
            let cond1 = rb.icmp(ICmpPred::SignedGt, x, ten);
            let outer = rb.gamma(
                cond1,
                state,
                &[x],
                |rb| {
                    let inner_x = rb.param(0);
                    let twenty = rb.const_i32(20);
                    let cond2 = rb.icmp(ICmpPred::SignedGt, inner_x, twenty);
                    let inner = rb.gamma(
                        cond2,
                        state,
                        &[inner_x],
                        |rb| BranchResult {
                            state,
                            values: vec![rb.const_i32(100)],
                        },
                        |rb| {
                            let v = rb.param(0);
                            let two = rb.const_i32(2);
                            let result =
                                rb.binary(BinaryOp::Mul, ArithFlags::default(), v, two, I32);
                            BranchResult {
                                state,
                                values: vec![result],
                            }
                        },
                    );
                    BranchResult {
                        state,
                        values: vec![inner.result(0)],
                    }
                },
                |rb| BranchResult {
                    state,
                    values: vec![rb.const_i32(0)],
                },
            );
            FnResult {
                state: outer.state,
                values: vec![outer.result(0)],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 30);
    }

    #[test]
    fn bubble_sort_array() {
        // Sort [4, 2, 7, 1, 3] using nested loops with store/load via GEP
        // Return first element (should be 1)
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let arr_ty_id = rvsdg.types.intern_array(ArrayType {
            element: I32,
            len: 5,
        });
        let arr_ty = TypeRef::Array(arr_ty_id);
        let elems: Vec<_> = [4, 2, 7, 1, 3]
            .iter()
            .map(|&v| rvsdg.constants.scalar(I32, ConstValue::Int(v)))
            .collect();
        let arr_const = rvsdg.constants.aggregate(arr_ty, &elems);
        let global_id = rvsdg.define_global(
            String::from("arr"),
            arr_ty,
            GlobalInit::Init(arr_const),
            false,
            GlobalLinkage::Internal,
        );
        let arr_ptr_ty = make_ptr_ty(&mut rvsdg, arr_ty);
        let i32_ptr_ty = make_ptr_ty(&mut rvsdg, I32);
        let func_id = rvsdg.declare_fn(String::from("test"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(func_id, |rb, state| {
            let arr_ptr = rb.global_ref(global_id, arr_ptr_ty);
            let zero = rb.const_i32(0);

            // Outer loop: i from 0 to 4
            let i_init = rb.const_i32(0);
            let outer = rb.theta(state, &[i_init], |rb| {
                let outer_i = rb.param(0);

                // Inner loop: j from 0 to 4-i (simplified: just 0 to 4)
                let j_init = rb.const_i32(0);
                let inner = rb.theta(state, &[j_init], |rb| {
                    let j = rb.param(0);
                    let one = rb.const_i32(1);
                    let j_plus_1 = rb.binary(BinaryOp::Add, ArithFlags::default(), j, one, I32);

                    let ptr_j = rb.ptr_offset(arr_ptr, arr_ty, &[zero, j], i32_ptr_ty, true);
                    let ptr_j1 =
                        rb.ptr_offset(arr_ptr, arr_ty, &[zero, j_plus_1], i32_ptr_ty, true);

                    let val_j = rb.load(state, ptr_j, I32, None, false);
                    let val_j1 = rb.load(val_j.state, ptr_j1, I32, None, false);

                    // if arr[j] > arr[j+1], swap
                    let should_swap = rb.icmp(ICmpPred::SignedGt, val_j.value, val_j1.value);
                    let swapped = rb.gamma(
                        should_swap,
                        val_j1.state,
                        &[val_j.value, val_j1.value],
                        |rb| {
                            // swap: store j1 value at j, j value at j+1
                            let a = rb.param(0);
                            let b = rb.param(1);
                            let s1 = rb.store(state, ptr_j, b, None, false);
                            let s2 = rb.store(s1, ptr_j1, a, None, false);
                            BranchResult {
                                state: s2,
                                values: vec![],
                            }
                        },
                        |_rb| BranchResult {
                            state,
                            values: vec![],
                        },
                    );

                    let four = rb.const_i32(4);
                    let cond = rb.icmp(ICmpPred::SignedLt, j_plus_1, four);
                    LoopResult {
                        condition: cond,
                        next_state: swapped.state,
                        next_vars: vec![j_plus_1],
                    }
                });

                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), outer_i, one, I32);
                let four = rb.const_i32(4);
                let cond = rb.icmp(ICmpPred::SignedLt, next_i, four);
                LoopResult {
                    condition: cond,
                    next_state: inner.state,
                    next_vars: vec![next_i],
                }
            });

            // Load first element (should be smallest = 1)
            let first_ptr = rb.ptr_offset(arr_ptr, arr_ty, &[zero, zero], i32_ptr_ty, true);
            let result = rb.load(outer.state, first_ptr, I32, None, false);
            FnResult {
                state: result.state,
                values: vec![result.value],
            }
        });
        assert_eq!(jit_run_i32(&rvsdg, "test"), 1);
    }
}

impl ScalarType {
    fn to_int_type<'a, 'ctx>(&'a self, context: &'ctx Context) -> IntType<'ctx> {
        match self {
            ScalarType::Bool => context.bool_type(),
            ScalarType::I8 => context.i8_type(),
            ScalarType::I16 => context.i16_type(),
            ScalarType::I32 => context.i32_type(),
            ScalarType::I64 => context.i64_type(),
            ScalarType::I128 => context.i128_type(),
            t => unreachable!("int type expected, got: {t:?}"),
        }
    }
    fn to_float_type<'a, 'ctx>(&'a self, context: &'ctx Context) -> FloatType<'ctx> {
        match self {
            ScalarType::F32 => context.f32_type(),
            ScalarType::F64 => context.f64_type(),
            t => unreachable!("float type expected, got: {t:?}"),
        }
    }
}
