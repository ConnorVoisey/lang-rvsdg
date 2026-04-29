use crate::{llvm_parser::instructions::Builder, rvsdg::ValueId};
use color_eyre::eyre::eyre;
use either::Either;
use llvm_ir::{Constant, Name, Operand};

/// If the operand is a direct global function reference, return its name.
/// Anything else (local SSA value, non-function global, expression, …) is an indirect call.
fn callee_as_global_name(operand: &Operand) -> Option<&Name> {
    match operand {
        Operand::ConstantOperand(cref) => match &**cref {
            Constant::GlobalReference { name, .. } => Some(name),
            _ => None,
        },
        _ => None,
    }
}

impl<'rb, 'g, 'm> Builder<'rb, 'g, 'm> {
    pub(super) fn call(&mut self, inst: &llvm_ir::instruction::Call) -> color_eyre::Result<()> {
        let callee_operand = match &inst.function {
            Either::Left(_inline_asm) => todo!("inline assembly call"),
            Either::Right(operand) => operand,
        };

        let args: Vec<ValueId> = inst
            .arguments
            .iter()
            .map(|(op, _attrs)| self.operand(op))
            .collect::<Result<_, _>>()?;

        let result = if let Some(name) = callee_as_global_name(callee_operand)
            && let Some(&fn_id) = self.rb.graph.fn_map.get(&name.to_string())
        {
            self.rb.call(fn_id, self.state, &args)
        } else {
            let callee_val = self.operand(callee_operand)?;
            let return_types = call_return_types(&inst.function_ty, self.llvm_mod, self.rb.graph)?;
            self.rb
                .call_indirect(callee_val, self.state, &args, &return_types)
        };

        self.state = result.state;

        if let Some(dest) = &inst.dest {
            match result.result_count {
                0 => {
                    return Err(eyre!("call has dest {dest:?} but callee returns no values"));
                }
                1 => {
                    self.name_to_val.insert(dest.clone(), result.first_result);
                }
                _ => todo!(
                    "multi-return call (LLVM struct return); decomposition into RVSDG return values not yet supported"
                ),
            }
        }
        Ok(())
    }
}

/// Extract the return-type list for an indirect call from its LLVM function type.
/// LLVM models a single return type (possibly void or a struct); RVSDG supports
/// multiple return values, but for the LLVM mapping we only ever produce 0 or 1.
fn call_return_types(
    function_ty: &llvm_ir::TypeRef,
    llvm_mod: &llvm_ir::Module,
    rvsdg_mod: &mut crate::rvsdg::RVSDGMod,
) -> color_eyre::Result<Vec<crate::rvsdg::types::TypeRef>> {
    let result_type = match &**function_ty {
        llvm_ir::Type::FuncType { result_type, .. } => result_type.clone(),
        ty => return Err(eyre!("call function_ty is not a FuncType, got {ty:?}")),
    };
    if matches!(&*result_type, llvm_ir::Type::VoidType) {
        return Ok(vec![]);
    }
    let ty = rvsdg_mod.types.convert_type_ref(&result_type, llvm_mod)?;
    Ok(vec![ty])
}
