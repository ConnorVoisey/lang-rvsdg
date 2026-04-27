use crate::{llvm_parser::instructions::Builder, rvsdg::ValueId};
use llvm_ir::types::Typed;

impl<'rb, 'g, 'm> Builder<'rb, 'g, 'm> {
    pub(super) fn extract_element(
        &mut self,
        inst: &llvm_ir::instruction::ExtractElement,
    ) -> color_eyre::Result<ValueId> {
        let vector = self.operand(&inst.vector)?;
        let index = self.operand(&inst.index)?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let element_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let val = self.rb.extract_lane(vector, index, element_type);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    pub(super) fn insert_element(
        &mut self,
        inst: &llvm_ir::instruction::InsertElement,
    ) -> color_eyre::Result<ValueId> {
        let vector = self.operand(&inst.vector)?;
        let element = self.operand(&inst.element)?;
        let index = self.operand(&inst.index)?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let vector_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let val = self.rb.insert_lane(vector, index, element, vector_type);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }
}
