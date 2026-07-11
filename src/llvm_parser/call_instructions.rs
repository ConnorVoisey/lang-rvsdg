use crate::{
    llvm_parser::{convert_calling_convention, convert_param_attrs, instructions::RegionLowerer},
    rvsdg::{
        State, ValueId,
        func::{Signature, SignatureId},
        types::TypeRef,
    },
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
    /// Intern the call SITE's ABI as one Signature: its function type,
    /// its per-ACTUAL-argument and return attributes, and its calling
    /// convention. The site is the source of truth for every call: for an
    /// indirect call the callee value is an opaque pointer so the ABI
    /// exists nowhere else, and for a direct VARIADIC call the variadic
    /// actual arguments' attributes (e.g. byval on a struct passed
    /// through `...`) exist nowhere else either -- the declaration has no
    /// parameter entries for them.
    fn call_site_signature(
        &mut self,
        inst: &llvm_ir::instruction::Call,
    ) -> color_eyre::Result<SignatureId> {
        let func_type = match self
            .rb
            .graph
            .types
            .convert_type_ref(&inst.function_ty, self.fn_ctx.llvm_mod)?
        {
            TypeRef::Func(id) => id,
            ty => return Err(eyre!("call function_ty is not a function type, got {ty:?}")),
        };
        let param_attrs = inst
            .arguments
            .iter()
            .map(|(_, attrs)| {
                convert_param_attrs(attrs, &mut self.rb.graph.types, self.fn_ctx.llvm_mod)
            })
            .collect::<color_eyre::Result<Vec<_>>>()?;
        let return_attrs = convert_param_attrs(
            &inst.return_attributes,
            &mut self.rb.graph.types,
            self.fn_ctx.llvm_mod,
        )?;
        Ok(self.rb.graph.signatures.intern(Signature {
            func_type,
            param_attrs,
            return_attrs,
            calling_convention: convert_calling_convention(inst.calling_convention)?,
        }))
    }

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
        let sig = self.call_site_signature(inst)?;

        // NOTE: the name must go through global_name_string -- llvm-ir's
        // Name Display prepends the % sigil, and a sigil-prefixed lookup
        // silently misses fn_map, demoting every direct call to the
        // indirect path (which works, the callee being a constant, but
        // bypasses the callee declaration's ABI and bloats the graph).
        let result = if let Some(name) = callee_as_global_name(callee_operand)
            && let Some(&fn_id) = self
                .rb
                .graph
                .fn_map
                .get(&crate::llvm_parser::global_name_string(name))
        {
            self.rb.call_with_signature(fn_id, state, &args, sig)
        } else {
            let callee_val = self.operand(callee_operand)?;
            self.rb.call_indirect(callee_val, state, &args, sig)
        };

        if let Some(dest) = &inst.dest {
            match result.result_count {
                0 => {
                    return Err(eyre!("call has dest {dest:?} but callee returns no values"));
                }
                1 => {
                    self.scopes.bind_name(dest, result.first_result);
                }
                _ => todo!(
                    "multi-return call (LLVM struct return); decomposition into RVSDG return values not yet supported"
                ),
            }
        }
        Ok(result.state)
    }
}
