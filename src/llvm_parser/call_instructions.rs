use crate::{
    llvm_parser::instructions::RegionLowerer,
    rvsdg::{State, ValueId, types::TypeRef},
};
use color_eyre::eyre::eyre;
use either::Either;
use llvm_ir::{Constant, Name, Operand};

/// If the operand is a direct global function reference, return its name.
/// Anything else (local SSA value, non-function global, expression, ...) is an indirect call.
fn callee_as_global_name(operand: &Operand) -> Option<&Name> {
    match operand {
        Operand::ConstantOperand(cref) => match &**cref {
            Constant::GlobalReference { name, .. } => Some(name),
            _ => None,
        },
        _ => None,
    }
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// Lower a Call instruction. Threads state through (calls are side-effecting).
    /// Returns the new state after the call completes.
    pub(super) fn call(
        &mut self,
        state: State,
        inst: &llvm_ir::instruction::Call,
    ) -> color_eyre::Result<State> {
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
            self.rb.call(fn_id, state, &args)
        } else {
            let callee_val = self.operand(callee_operand)?;
            // The call site's function type is the only place the callee's
            // signature exists (the callee value is an opaque pointer), so
            // it is interned and stored on the CallIndirect node.
            let fn_ty = match self
                .rb
                .graph
                .types
                .convert_type_ref(&inst.function_ty, self.fn_ctx.llvm_mod)?
            {
                TypeRef::Func(id) => id,
                ty => return Err(eyre!("call function_ty is not a function type, got {ty:?}")),
            };
            self.rb.call_indirect(callee_val, state, &args, fn_ty)
        };

        if let Some(dest) = &inst.dest {
            match result.result_count {
                0 => {
                    return Err(eyre!("call has dest {dest:?} but callee returns no values"));
                }
                1 => {
                    self.scopes.bind_name(dest.clone(), result.first_result);
                }
                _ => todo!(
                    "multi-return call (LLVM struct return); decomposition into RVSDG return values not yet supported"
                ),
            }
        }
        Ok(result.state)
    }
}
