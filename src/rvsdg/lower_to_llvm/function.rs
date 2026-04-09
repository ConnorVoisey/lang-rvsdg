use crate::rvsdg::{
    FuncId, GlobalId, GlobalInit, GlobalLinkage, RVSDGMod, Region, ValueId, ValueKind,
    func::{FnLinkageType, Function},
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
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

impl RVSDGMod {
    pub(crate) fn register_fn<'a, 'ctx>(
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

    pub(crate) fn lower_fn<'a, 'ctx>(
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
}
