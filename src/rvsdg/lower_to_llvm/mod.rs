use crate::rvsdg::{
    FuncId, GlobalId, GlobalInit, Linkage, RVSDGMod, Region, ValueId, ValueKind,
    func::Function,
    types::{ScalarType, TypeRef},
};
use inkwell::{
    AddressSpace, OptimizationLevel,
    builder::{Builder, BuilderError},
    context::Context,
    module::Module,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
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

impl Linkage {
    fn to_llvm(&self) -> inkwell::module::Linkage {
        match self {
            Linkage::Internal => inkwell::module::Linkage::Internal,
            Linkage::External => inkwell::module::Linkage::External,
            Linkage::LinkOnce => inkwell::module::Linkage::LinkOnceAny,
            Linkage::LinkOnceODR => inkwell::module::Linkage::LinkOnceODR,
            Linkage::Weak => inkwell::module::Linkage::WeakAny,
            Linkage::WeakODR => inkwell::module::Linkage::WeakODR,
            Linkage::AvailableExternally => inkwell::module::Linkage::AvailableExternally,
        }
    }
}
