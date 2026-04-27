use crate::{
    llvm_parser::strongly_connected_components::Scc,
    rvsdg::{
        ArithFlags, BinaryOp, CastOp, FCmpPred, ICmpPred, MemoryOrdering, State, UnaryOp, ValueId,
        builder::RegionBuilder,
    },
};
use llvm_ir::{
    FPPredicate, Function, Instruction, IntPredicate, Module, Name, Operand,
    instruction::{HasResult, MemoryOrdering as LlvmMemoryOrdering},
    types::Typed,
};
use rustc_hash::FxHashMap;

fn convert_int_pred(p: IntPredicate) -> ICmpPred {
    match p {
        IntPredicate::EQ => ICmpPred::Eq,
        IntPredicate::NE => ICmpPred::Ne,
        IntPredicate::UGT => ICmpPred::UnsignedGt,
        IntPredicate::UGE => ICmpPred::UnsignedGe,
        IntPredicate::ULT => ICmpPred::UnsignedLt,
        IntPredicate::ULE => ICmpPred::UnsignedLe,
        IntPredicate::SGT => ICmpPred::SignedGt,
        IntPredicate::SGE => ICmpPred::SignedGe,
        IntPredicate::SLT => ICmpPred::SignedLt,
        IntPredicate::SLE => ICmpPred::SignedLe,
    }
}

fn convert_mem_ordering(o: LlvmMemoryOrdering) -> MemoryOrdering {
    match o {
        LlvmMemoryOrdering::Unordered
        | LlvmMemoryOrdering::Monotonic
        | LlvmMemoryOrdering::NotAtomic => MemoryOrdering::Relaxed,
        LlvmMemoryOrdering::Acquire => MemoryOrdering::Acquire,
        LlvmMemoryOrdering::Release => MemoryOrdering::Release,
        LlvmMemoryOrdering::AcquireRelease => MemoryOrdering::AcquireRelease,
        LlvmMemoryOrdering::SequentiallyConsistent => MemoryOrdering::SequentiallyConsistent,
    }
}

fn convert_fp_pred(p: FPPredicate) -> FCmpPred {
    match p {
        FPPredicate::False => FCmpPred::False,
        FPPredicate::OEQ => FCmpPred::OrderedEq,
        FPPredicate::OGT => FCmpPred::OrderedGt,
        FPPredicate::OGE => FCmpPred::OrderedGe,
        FPPredicate::OLT => FCmpPred::OrderedLt,
        FPPredicate::OLE => FCmpPred::OrderedLe,
        FPPredicate::ONE => FCmpPred::OrderedNe,
        FPPredicate::ORD => FCmpPred::Ordered,
        FPPredicate::UNO => FCmpPred::Unordered,
        FPPredicate::UEQ => FCmpPred::UnorderedEq,
        FPPredicate::UGT => FCmpPred::UnorderedGt,
        FPPredicate::UGE => FCmpPred::UnorderedGe,
        FPPredicate::ULT => FCmpPred::UnorderedLt,
        FPPredicate::ULE => FCmpPred::UnorderedLe,
        FPPredicate::UNE => FCmpPred::UnorderedNe,
        FPPredicate::True => FCmpPred::True,
    }
}

pub struct Builder<'rb, 'g, 'm> {
    pub rb: &'rb mut RegionBuilder<'g>,
    pub llvm_mod: &'m Module,
    pub state: State,
    pub name_to_val: FxHashMap<Name, ValueId>,
}

impl<'rb, 'g, 'm> Builder<'rb, 'g, 'm> {
    pub fn new(rb: &'rb mut RegionBuilder<'g>, state: State, llvm_mod: &'m Module) -> Self {
        Self {
            rb,
            llvm_mod,
            state,
            name_to_val: FxHashMap::default(),
        }
    }

    pub fn lower_scc(&mut self, func: &Function, scc: &Scc) -> color_eyre::Result<()> {
        for block_id in &scc.blocks {
            let bb = &func.basic_blocks[block_id.0 as usize];
            for inst in &bb.instrs {
                self.lower_instruction(inst)?;
            }
        }
        Ok(())
    }

    fn lower_instruction(&mut self, inst: &Instruction) -> color_eyre::Result<()> {
        match inst {
            Instruction::Add(i) => {
                self.binary(i, BinaryOp::Add, ArithFlags::wrap(i.nsw, i.nuw))?;
            }
            Instruction::Sub(i) => {
                self.binary(i, BinaryOp::Sub, ArithFlags::wrap(i.nsw, i.nuw))?;
            }
            Instruction::Mul(i) => {
                self.binary(i, BinaryOp::Mul, ArithFlags::wrap(i.nsw, i.nuw))?;
            }
            Instruction::UDiv(i) => {
                self.binary(i, BinaryOp::UnsignedDiv, ArithFlags::exact(i.exact))?;
            }
            Instruction::SDiv(i) => {
                self.binary(i, BinaryOp::SignedDiv, ArithFlags::exact(i.exact))?;
            }
            Instruction::URem(i) => {
                self.binary(i, BinaryOp::UnsignedRem, ArithFlags::default())?;
            }
            Instruction::SRem(i) => {
                self.binary(i, BinaryOp::SignedRem, ArithFlags::default())?;
            }
            Instruction::And(i) => {
                self.binary(i, BinaryOp::And, ArithFlags::default())?;
            }
            Instruction::Or(i) => {
                self.binary(i, BinaryOp::Or, ArithFlags::default())?;
            }
            Instruction::Xor(i) => {
                self.binary(i, BinaryOp::Xor, ArithFlags::default())?;
            }
            Instruction::Shl(i) => {
                self.binary(i, BinaryOp::ShiftLeft, ArithFlags::wrap(i.nsw, i.nuw))?;
            }
            Instruction::LShr(i) => {
                self.binary(i, BinaryOp::LogicalShiftRight, ArithFlags::exact(i.exact))?;
            }
            Instruction::AShr(i) => {
                self.binary(i, BinaryOp::ArithShiftRight, ArithFlags::exact(i.exact))?;
            }
            Instruction::FAdd(i) => {
                self.binary(i, BinaryOp::FloatAdd, ArithFlags::default())?;
            }
            Instruction::FSub(i) => {
                self.binary(i, BinaryOp::FloatSub, ArithFlags::default())?;
            }
            Instruction::FMul(i) => {
                self.binary(i, BinaryOp::FloatMul, ArithFlags::default())?;
            }
            Instruction::FDiv(i) => {
                self.binary(i, BinaryOp::FloatDiv, ArithFlags::default())?;
            }
            Instruction::FRem(i) => {
                self.binary(i, BinaryOp::FloatRem, ArithFlags::default())?;
            }
            Instruction::FNeg(i) => {
                self.unary(i, UnaryOp::FloatNeg)?;
            }
            Instruction::ExtractElement(i) => {
                self.extract_element(i)?;
            }
            Instruction::InsertElement(i) => {
                self.insert_element(i)?;
            }
            Instruction::ShuffleVector(_) => {
                // Mask is a constant vector — needs per-element decomposition.
                // Skipped until vector-constant lowering and undef-element handling are decided.
                todo!("shufflevector")
            }
            Instruction::ExtractValue(i) => {
                self.extract_value(i)?;
            }
            Instruction::InsertValue(i) => {
                self.insert_value(i)?;
            }
            Instruction::Alloca(i) => {
                self.alloca(i)?;
            }
            Instruction::Load(i) => {
                self.load(i)?;
            }
            Instruction::Store(i) => {
                self.store(i)?;
            }
            Instruction::Fence(i) => {
                self.fence(i);
            }
            Instruction::CmpXchg(cmp_xchg) => todo!(),
            Instruction::AtomicRMW(atomic_rmw) => todo!(),
            Instruction::GetElementPtr(i) => {
                self.get_element_ptr(i)?;
            }
            Instruction::Trunc(i) => {
                self.cast(i, CastOp::Truncate)?;
            }
            Instruction::ZExt(i) => {
                self.cast(i, CastOp::ZeroExtend)?;
            }
            Instruction::SExt(i) => {
                self.cast(i, CastOp::SignExtend)?;
            }
            Instruction::FPTrunc(i) => {
                self.cast(i, CastOp::FloatTruncate)?;
            }
            Instruction::FPExt(i) => {
                self.cast(i, CastOp::FloatExtend)?;
            }
            Instruction::FPToUI(i) => {
                self.cast(i, CastOp::FloatToUnsigned)?;
            }
            Instruction::FPToSI(i) => {
                self.cast(i, CastOp::FloatToSigned)?;
            }
            Instruction::UIToFP(i) => {
                self.cast(i, CastOp::UnsignedToFloat)?;
            }
            Instruction::SIToFP(i) => {
                self.cast(i, CastOp::SignedToFloat)?;
            }
            Instruction::PtrToInt(i) => {
                self.cast(i, CastOp::PtrToInt)?;
            }
            Instruction::IntToPtr(i) => {
                self.cast(i, CastOp::IntToPtr)?;
            }
            Instruction::BitCast(i) => {
                self.cast(i, CastOp::Bitcast)?;
            }
            Instruction::AddrSpaceCast(i) => {
                self.cast(i, CastOp::Bitcast)?;
            }
            Instruction::ICmp(i) => {
                self.icmp(i)?;
            }
            Instruction::FCmp(i) => {
                self.fcmp(i)?;
            }
            Instruction::Phi(phi) => todo!(),
            Instruction::Select(i) => {
                self.select(i)?;
            }
            Instruction::Freeze(i) => {
                self.freeze(i)?;
            }
            Instruction::Call(call) => todo!(),
            Instruction::VAArg(vaarg) => todo!(),
            Instruction::LandingPad(landing_pad) => todo!(),
            Instruction::CatchPad(catch_pad) => todo!(),
            Instruction::CleanupPad(cleanup_pad) => todo!(),
        }
        Ok(())
    }

    fn binary<I>(
        &mut self,
        inst: &I,
        op: BinaryOp,
        flags: ArithFlags,
    ) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::BinaryOp + HasResult + Typed,
    {
        let left = self.operand(inst.get_operand0())?;
        let right = self.operand(inst.get_operand1())?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let dest = inst.get_result().clone();
        let val = self.rb.binary(op, flags, left, right, ty);
        self.name_to_val.insert(dest, val);
        Ok(val)
    }

    fn unary<I>(&mut self, inst: &I, op: UnaryOp) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::UnaryOp + HasResult + Typed,
    {
        let operand = self.operand(inst.get_operand())?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let dest = inst.get_result().clone();
        let val = self.rb.unary(op, operand, ty);
        self.name_to_val.insert(dest, val);
        Ok(val)
    }

    fn cast<I>(&mut self, inst: &I, op: CastOp) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::UnaryOp + HasResult + Typed,
    {
        let operand = self.operand(inst.get_operand())?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let dest = inst.get_result().clone();
        let val = self.rb.cast(op, operand, ty);
        self.name_to_val.insert(dest, val);
        Ok(val)
    }

    fn extract_value(
        &mut self,
        inst: &llvm_ir::instruction::ExtractValue,
    ) -> color_eyre::Result<ValueId> {
        let aggregate = self.operand(&inst.aggregate)?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let field_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let val = self.rb.extract_field(aggregate, &inst.indices, field_type);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn insert_value(
        &mut self,
        inst: &llvm_ir::instruction::InsertValue,
    ) -> color_eyre::Result<ValueId> {
        let aggregate = self.operand(&inst.aggregate)?;
        let element = self.operand(&inst.element)?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let aggregate_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let val = self
            .rb
            .insert_field(aggregate, element, &inst.indices, aggregate_type);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn get_element_ptr(
        &mut self,
        inst: &llvm_ir::instruction::GetElementPtr,
    ) -> color_eyre::Result<ValueId> {
        let base = self.operand(&inst.address)?;
        let base_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&inst.source_element_type, self.llvm_mod)?;
        let indices = inst
            .indices
            .iter()
            .map(|op| self.operand(op))
            .collect::<Result<Vec<_>, _>>()?;
        let llvm_result_ty = inst.get_type(&self.llvm_mod.types);
        let result_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_result_ty, self.llvm_mod)?;
        let val = self
            .rb
            .ptr_offset(base, base_type, &indices, result_type, inst.in_bounds);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn icmp(&mut self, inst: &llvm_ir::instruction::ICmp) -> color_eyre::Result<ValueId> {
        let left = self.operand(&inst.operand0)?;
        let right = self.operand(&inst.operand1)?;
        let pred = convert_int_pred(inst.predicate);
        let val = self.rb.icmp(pred, left, right);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn fcmp(&mut self, inst: &llvm_ir::instruction::FCmp) -> color_eyre::Result<ValueId> {
        let left = self.operand(&inst.operand0)?;
        let right = self.operand(&inst.operand1)?;
        let pred = convert_fp_pred(inst.predicate);
        let val = self.rb.fcmp(pred, left, right);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn load(&mut self, inst: &llvm_ir::instruction::Load) -> color_eyre::Result<ValueId> {
        let addr = self.operand(&inst.address)?;
        let loaded_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&inst.loaded_ty, self.llvm_mod)?;
        let align = (inst.alignment != 0).then_some(inst.alignment);

        let result = match &inst.atomicity {
            Some(at) => self.rb.atomic_load(
                self.state,
                addr,
                loaded_type,
                convert_mem_ordering(at.mem_ordering),
                align,
            ),
            None => self
                .rb
                .load(self.state, addr, loaded_type, align, inst.volatile),
        };
        self.state = result.state;
        self.name_to_val.insert(inst.dest.clone(), result.value);
        Ok(result.value)
    }

    fn store(&mut self, inst: &llvm_ir::instruction::Store) -> color_eyre::Result<()> {
        let addr = self.operand(&inst.address)?;
        let value = self.operand(&inst.value)?;
        let align = (inst.alignment != 0).then_some(inst.alignment);

        self.state = match &inst.atomicity {
            Some(at) => self.rb.atomic_store(
                self.state,
                addr,
                value,
                convert_mem_ordering(at.mem_ordering),
                align,
            ),
            None => self.rb.store(self.state, addr, value, align, inst.volatile),
        };
        Ok(())
    }

    fn alloca(&mut self, inst: &llvm_ir::instruction::Alloca) -> color_eyre::Result<ValueId> {
        let count = self.operand(&inst.num_elements)?;
        let elem_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&inst.allocated_type, self.llvm_mod)?;
        let llvm_ptr_ty = inst.get_type(&self.llvm_mod.types);
        let ptr_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ptr_ty, self.llvm_mod)?;

        let result = self.rb.alloca(self.state, elem_type, count, ptr_type);
        self.state = result.state;
        self.name_to_val.insert(inst.dest.clone(), result.ptr);
        Ok(result.ptr)
    }

    fn fence(&mut self, inst: &llvm_ir::instruction::Fence) {
        let ordering = convert_mem_ordering(inst.atomicity.mem_ordering);
        self.state = self.rb.fence(self.state, ordering);
    }

    fn select(&mut self, inst: &llvm_ir::instruction::Select) -> color_eyre::Result<ValueId> {
        let cond = self.operand(&inst.condition)?;
        let t = self.operand(&inst.true_value)?;
        let f = self.operand(&inst.false_value)?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let val = self.rb.ternary(cond, t, f, ty);
        self.name_to_val.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn freeze<I>(&mut self, inst: &I) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::UnaryOp + HasResult + Typed,
    {
        let operand = self.operand(inst.get_operand())?;
        let llvm_ty = inst.get_type(&self.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.llvm_mod)?;
        let dest = inst.get_result().clone();
        let val = self.rb.freeze(operand, ty);
        self.name_to_val.insert(dest, val);
        Ok(val)
    }

    pub(super) fn operand(&mut self, op: &Operand) -> color_eyre::Result<ValueId> {
        match op {
            Operand::LocalOperand { name, .. } => Ok(*self
                .name_to_val
                .get(name)
                .expect("ssa value should already have been defined")),
            Operand::ConstantOperand(constant_ref) => {
                let const_id = self
                    .rb
                    .graph
                    .convert_const_ref(constant_ref.clone(), self.llvm_mod)?;
                let ty = self.rb.graph.constants.get(const_id).ty;
                Ok(self.rb.const_pool_ref(const_id, ty))
            }
            Operand::MetadataOperand => {
                todo!("MetadataOperand is currently unsupported within llvm_ir")
            }
        }
    }
}
