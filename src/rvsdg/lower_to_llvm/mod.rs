use crate::rvsdg::{
    FuncId, GlobalId, GlobalInit, Linkage, RVSDGMod, Region, ValueId, ValueKind,
    func::Function,
    types::{ScalarType, TypeRef},
};
use color_eyre::eyre::{bail, eyre};
use inkwell::{
    AddressSpace, OptimizationLevel,
    builder::Builder,
    context::Context,
    module::Module,
    targets::{CodeModel, FileType, RelocMode, Target, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum, FunctionValue, GlobalValue},
};
use std::{path::Path, process::Command};

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
    #[tracing::instrument(skip_all)]
    pub fn lower_to_llvm_module<'ctx>(
        &self,
        context: &'ctx Context,
    ) -> color_eyre::Result<Module<'ctx>> {
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

    pub fn output_with_llvm(
        &self,
        output: &str,
        link_inputs: &[String],
        include_dirs: &[String],
        quiet: bool,
    ) -> color_eyre::Result<()> {
        // initialise things (guarded so concurrent callers don't race the
        // process-global target registry)
        crate::init_llvm_native()?;

        let context = Context::create();
        let module = self.lower_to_llvm_module(&context)?;
        if !quiet {
            eprintln!("LLVM IR:");
            eprintln!("{}", module.print_to_string().to_string());
        }

        // more output things
        let llvm_triple = TargetTriple::create(&self.target.to_string());
        let target = Target::from_triple(&llvm_triple)
            .map_err(|e| eyre!("failed to get target for triple {}: {e}", self.target))?;

        // This opt level is the LLVM *CodeGenOptLevel* only -- `write_to_file`
        // runs the backend pipeline (instruction selection, scheduling,
        // register allocation, machine-level opts), never the mid-level IR
        // optimizer (instcombine/GVN/LICM/vectorize), which is only run via a
        // PassBuilder pipeline we deliberately don't invoke. So the mid level
        // stays "-O0" (RVSDG owns those optimizations); `Aggressive` (-O3)
        // gives the greedy register allocator and aggressive instruction
        // selection/scheduling for the final code, not extra IR optimization.
        let machine = target
            .create_target_machine(
                &llvm_triple,
                "generic",
                "",
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| eyre!("failed to create target machine for triple {}", self.target))?;

        let obj_file = format!("{}.o", output);
        let obj_path = Path::new(&obj_file);
        machine
            .write_to_file(&module, FileType::Object, obj_path)
            .map_err(|e| eyre!("failed to write object file {}: {e}", obj_path.display()))?;

        // Status/diagnostics go to stderr so the compiler never writes to
        // stdout -- that belongs to the compiled program when it runs.
        eprintln!("Wrote object file: {}", obj_path.display());

        let obj_arg = obj_path
            .to_str()
            .ok_or_else(|| eyre!("object path {} is not valid UTF-8", obj_path.display()))?;

        // Link the compiled object together with any extra inputs (e.g. a
        // benchmark harness like PolyBench's `utilities/polybench.c`). `cc`
        // compiles any `.c` inputs and links everything; the `-I` paths are
        // passed so those sources can find their headers.
        let mut link = Command::new("cc");
        link.arg(obj_arg);
        for input in link_inputs {
            link.arg(input);
        }
        for dir in include_dirs {
            link.arg("-I").arg(dir);
        }
        link.args(["-o", output]);
        let status = link
            .status()
            .map_err(|e| eyre!("failed to invoke linker (cc): {e}"))?;

        if !status.success() {
            bail!("linking failed with status: {status}");
        }
        eprintln!("Linked executable: ./{output}");
        eprintln!("Run it with:  ./{output}");
        Ok(())
    }

    fn lower_mod<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
    ) -> color_eyre::Result<()> {
        // For now we'll use a naive implementation that converts the RVSDG directly to llvm
        // without using predicates.
        // TODO: replace this implemenation with this https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43246.pdf

        self.lower_globals(llvm_builder, mapper)?;
        for func in self.functions.iter() {
            self.register_fn(llvm_builder, mapper, func)?;
        }
        for func in self.functions.iter() {
            if func.lambda_val.is_none() {
                continue; // declaration only, no body to lower
            }
            self.lower_fn(llvm_builder, mapper, func)?;
        }
        if let Err(e) = llvm_builder.module.verify() {
            bail!("LLVM module verification failed: {e}");
        }

        Ok(())
    }

    fn lower_globals<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
    ) -> color_eyre::Result<()> {
        // Pass 1: declare every global. This must finish before any
        // initializer is lowered, because an initializer can reference another
        // global that is declared later (e.g. `@a = ptr @b` with `@b` below
        // `@a`); that GlobalAddr resolves through the mapper, which only has
        // the global once it's declared here. Same two-pass shape as the
        // frontend's `from_llvm_mod`.
        for (i, global) in self.globals.iter().enumerate() {
            let llvm_type = self.type_to_basic_type_llvm(llvm_builder.context, global.ty)?;
            let glob = llvm_builder
                .module
                // TODO: replace the address space with the RVSDG address space,
                // Inkwell stores this as an i16, we store as a string
                .add_global(llvm_type, None, &global.name);
            glob.set_constant(global.is_constant);
            glob.set_linkage(global.linkage.to_llvm());
            mapper.set_global(GlobalId(i as u32), glob);
        }

        // Pass 2: set initializers, now that every global resolves.
        for (i, global) in self.globals.iter().enumerate() {
            if let GlobalInit::Init(const_id) = global.initializer {
                let const_val = self.lower_const_id(llvm_builder, mapper, const_id)?;
                mapper
                    .get_global(GlobalId(i as u32))
                    .ok_or_else(|| eyre!("global {i} was not declared in pass 1"))?
                    .set_initializer(&const_val as &dyn BasicValue);
            }
        }
        Ok(())
    }

    fn register_fn<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
    ) -> color_eyre::Result<()> {
        if rvsdg_func.return_types.len() >= 2 {
            bail!(
                "function `{}` has {} return values; LLVM supports at most one",
                rvsdg_func.name,
                rvsdg_func.return_types.len()
            );
        }

        let param_types = rvsdg_func
            .params
            .iter()
            .map(|param| self.type_to_basic_meta_llvm(llvm_builder.context, param.ty))
            .collect::<color_eyre::Result<Vec<_>>>()?;
        let llvm_fn_type = if let Some(&ret_ty) = rvsdg_func.return_types.first() {
            self.type_to_basic_type_llvm(llvm_builder.context, ret_ty)?
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
        Ok(())
    }

    #[tracing::instrument(skip_all, fields(func = %rvsdg_func.name))]
    fn lower_fn<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
    ) -> color_eyre::Result<()> {
        let func = mapper.get_fn(rvsdg_func.id).ok_or_else(|| {
            eyre!(
                "function `{}` was not registered before lowering its body",
                rvsdg_func.name
            )
        })?;
        let entry = llvm_builder.context.append_basic_block(func, "entry");
        llvm_builder.builder.position_at_end(entry);
        let fn_val = rvsdg_func.lambda_val.ok_or_else(|| {
            eyre!(
                "function `{}` has no lambda value set during RVSDG construction",
                rvsdg_func.name
            )
        })?;
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
                    let param = func.get_nth_param(i).ok_or_else(|| {
                        eyre!("function `{}` is missing parameter {i}", rvsdg_func.name)
                    })?;
                    mapper.set_val(param_id, param);
                }

                self.lower_region(llvm_builder, mapper, rvsdg_func, region)?;

                // regions results should be added from inside lower_region
                let res = self.value_pool.get(region.results);
                match res.len() {
                    0 => llvm_builder.builder.build_return(None)?,
                    1 => {
                        let val = mapper.get_val(res[0]).ok_or_else(|| {
                            eyre!("return value of `{}` was not lowered", rvsdg_func.name)
                        })?;
                        llvm_builder
                            .builder
                            .build_return(Some(&val as &dyn BasicValue))?
                    }
                    n => bail!(
                        "function `{}` returns {n} values; LLVM supports at most one",
                        rvsdg_func.name
                    ),
                }
            }
            t => bail!(
                "function `{}` lambda has unexpected value kind {t:?}",
                rvsdg_func.name
            ),
        };
        Ok(())
    }

    fn lower_region<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        region: &Region,
    ) -> color_eyre::Result<()> {
        for &value_id in region.nodes.iter() {
            self.lower_value(llvm_builder, mapper, rvsdg_func, value_id)?;
        }
        Ok(())
    }
    fn type_to_basic_type_llvm<'b>(
        &self,
        context: &'b Context,
        ty: TypeRef,
    ) -> color_eyre::Result<BasicTypeEnum<'b>> {
        let basic = match ty {
            TypeRef::State => bail!("`state` is an IR-only type with no LLVM basic type"),
            TypeRef::Scalar(scalar_type) => match scalar_type {
                ScalarType::Bool => BasicTypeEnum::IntType(context.bool_type()),
                ScalarType::I8 => BasicTypeEnum::IntType(context.i8_type()),
                ScalarType::I16 => BasicTypeEnum::IntType(context.i16_type()),
                ScalarType::I32 => BasicTypeEnum::IntType(context.i32_type()),
                ScalarType::I64 => BasicTypeEnum::IntType(context.i64_type()),
                ScalarType::I128 => BasicTypeEnum::IntType(context.i128_type()),
                ScalarType::F32 => BasicTypeEnum::FloatType(context.f32_type()),
                ScalarType::F64 => BasicTypeEnum::FloatType(context.f64_type()),
                // Void is not a BasicType in LLVM -- it only appears as a function
                // return type, never as a value/parameter/alloca type.
                ScalarType::Void => bail!("`void` is not a basic type"),
            },
            TypeRef::Ptr(_) => {
                BasicTypeEnum::PointerType(context.ptr_type(AddressSpace::default()))
            }
            TypeRef::Array(array_type_id) => {
                let arr = self.types.get_array(array_type_id);
                let elem = self.type_to_basic_type_llvm(context, arr.element)?;
                BasicTypeEnum::ArrayType(elem.array_type(arr.len as u32))
            }
            TypeRef::Struct(struct_id) => {
                let def = self.types.get_struct(struct_id);
                let field_types: Vec<BasicTypeEnum> = def
                    .fields
                    .iter()
                    .map(|f| self.type_to_basic_type_llvm(context, f.field_type))
                    .collect::<color_eyre::Result<_>>()?;
                BasicTypeEnum::StructType(context.struct_type(&field_types, false))
            }
            TypeRef::Vector(vector_type_id) => {
                let vec = self.types.get_vector(vector_type_id);
                let elem = self.type_to_basic_type_llvm(context, vec.element)?;
                match elem {
                    BasicTypeEnum::IntType(t) => BasicTypeEnum::VectorType(t.vec_type(vec.lanes)),
                    BasicTypeEnum::FloatType(t) => BasicTypeEnum::VectorType(t.vec_type(vec.lanes)),
                    BasicTypeEnum::PointerType(t) => {
                        BasicTypeEnum::VectorType(t.vec_type(vec.lanes))
                    }
                    _ => bail!("vector element must be a scalar or pointer type"),
                }
            }
            // FuncType is not a BasicType -- functions exist only as pointers
            // (opaque ptr in LLVM 17+). If a TypeRef::Func reaches here, the
            // caller has a bug.
            TypeRef::Func(_) => bail!("function type is not a basic type"),
            // A control/predicate value is an alternative index; lower it to an
            // `i32`. This is what makes the gamma backend take its switch path
            // (value k -> region k) for predicate-driven gammas.
            TypeRef::Control(_) => BasicTypeEnum::IntType(context.i32_type()),
        };
        Ok(basic)
    }

    /// A parameter/argument-position type. Identical to
    /// [`type_to_basic_type_llvm`](Self::type_to_basic_type_llvm) except for the
    /// wrapper enum, so it just converts that result -- every `BasicTypeEnum`
    /// has a `BasicMetadataTypeEnum` counterpart.
    fn type_to_basic_meta_llvm<'b>(
        &self,
        context: &'b Context,
        ty: TypeRef,
    ) -> color_eyre::Result<BasicMetadataTypeEnum<'b>> {
        Ok(self.type_to_basic_type_llvm(context, ty)?.into())
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
