use crate::rvsdg::{
    RVSDGMod, UnaryOp, ValueId,
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
    #[inline]
    pub(crate) fn lower_unary<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        mapper: &mut ValueMapper<'ctx>,
        rvsdg_func: &Function,
        op: UnaryOp,
        operand: ValueId,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let val = self.expect_value(llvm_builder, mapper, rvsdg_func, operand)?;
        let b = &llvm_builder.builder;
        let val_int = || val.into_int_value();
        let val_float = || val.into_float_value();

        Ok(match op {
            UnaryOp::BitNot => BasicValueEnum::IntValue(b.build_not(val_int(), "not")?),
            UnaryOp::FloatNeg => {
                BasicValueEnum::FloatValue(b.build_float_neg(val_float(), "fneg")?)
            }
            UnaryOp::CountLeadingZeros => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                let is_zero_poison = llvm_builder.context.bool_type().const_zero();
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ctlz",
                    &[int_type],
                    &[int_val.into(), is_zero_poison.into()],
                    "ctlz",
                )
            }
            UnaryOp::CountTrailingZeros => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                let is_zero_poison = llvm_builder.context.bool_type().const_zero();
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.cttz",
                    &[int_type],
                    &[int_val.into(), is_zero_poison.into()],
                    "cttz",
                )
            }
            UnaryOp::CountOnes => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ctpop",
                    &[int_type],
                    &[int_val.into()],
                    "ctpop",
                )
            }
            UnaryOp::ByteSwap => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.bswap",
                    &[int_type],
                    &[int_val.into()],
                    "bswap",
                )
            }
            UnaryOp::BitReverse => {
                let int_val = val_int();
                let int_type = BasicTypeEnum::IntType(int_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.bitreverse",
                    &[int_type],
                    &[int_val.into()],
                    "bitrev",
                )
            }
            UnaryOp::FloatAbs => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.fabs",
                    &[float_type],
                    &[float_val.into()],
                    "fabs",
                )
            }
            UnaryOp::FloatFloor => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.floor",
                    &[float_type],
                    &[float_val.into()],
                    "floor",
                )
            }
            UnaryOp::FloatCeil => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.ceil",
                    &[float_type],
                    &[float_val.into()],
                    "ceil",
                )
            }
            UnaryOp::FloatRound => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.round",
                    &[float_type],
                    &[float_val.into()],
                    "round",
                )
            }
            UnaryOp::FloatSqrt => {
                let float_val = val_float();
                let float_type = BasicTypeEnum::FloatType(float_val.get_type());
                self.call_overloaded_intrinsic(
                    llvm_builder,
                    "llvm.sqrt",
                    &[float_type],
                    &[float_val.into()],
                    "sqrt",
                )
            }
        })
    }

    fn call_overloaded_intrinsic<'a, 'ctx>(
        &self,
        llvm_builder: &LLVMBuilderCtx<'a, 'ctx>,
        intrinsic_name: &str,
        param_types: &[BasicTypeEnum<'ctx>],
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let intrinsic = Intrinsic::find(intrinsic_name)
            .unwrap_or_else(|| panic!("intrinsic {intrinsic_name} not found"));
        let func = intrinsic
            .get_declaration(llvm_builder.module, param_types)
            .unwrap_or_else(|| panic!("failed to get declaration for {intrinsic_name}"));
        match llvm_builder
            .builder
            .build_call(func, args, name)
            .expect("failed to build intrinsic call")
            .try_as_basic_value()
        {
            ValueKind::Basic(val) => val,
            ValueKind::Instruction(_) => {
                panic!("intrinsic {intrinsic_name} returned void, expected a value")
            }
        }
    }
}
