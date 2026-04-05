use crate::rvsdg::{
    ConstId, ConstValue, ConstantDef, ConstantKind, RVSDGMod,
    lower_to_llvm::{LLVMBuilderCtx, ValueMapper},
    types::TypeRef,
};
use inkwell::{AddressSpace, values::BasicValueEnum};

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
            ConstantKind::Zero => todo!(),
            ConstantKind::Aggregate(const_ids_span) => todo!(),
            ConstantKind::String(items) => {
                BasicValueEnum::ArrayValue(llvm_builder.context.const_string(items, false))
            }
            ConstantKind::GlobalAddr(global_id) => todo!(),
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
            ConstValue::F32(val) => {
                BasicValueEnum::FloatValue(llvm_builder.context.f32_type().const_float(*val as f64))
            }
            ConstValue::F64(val) => {
                BasicValueEnum::FloatValue(llvm_builder.context.f64_type().const_float(*val as f64))
            }
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
