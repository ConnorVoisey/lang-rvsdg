use crate::{
    llvm_parser::{FnCtx, block_mapper::BasicBlockId, control_flow::scopes::SymbolScopes},
    rvsdg::{
        ArithFlags, AtomicRMWOp, BinaryOp, CastOp, FCmpPred, ICmpPred, MatchArm, MemoryOrdering,
        State, UnaryOp, ValueId,
        builder::{AllocaResult, LoadResult, RegionBuilder},
    },
};
use llvm_ir::{
    FPPredicate, Instruction, IntPredicate, Operand,
    instruction::{HasResult, MemoryOrdering as LlvmMemoryOrdering},
    types::Typed,
};

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

/// NOTE: only the ordering half of llvm-ir's `Atomicity` is converted;
/// the synchronisation SCOPE (`synch_scope`) is deliberately dropped, so
/// singlethread atomics and fences (`atomic_signal_fence`) are lowered at
/// full system scope. That is a correct strengthening -- a system-scope
/// operation orders everything a singlethread one would -- but emits
/// unnecessary hardware fences on ARM. Restoring it needs a scope field
/// on the atomic ValueKinds plus, at lowering, build_fence's second
/// argument (already a singlethread flag, currently hardcoded 0) and one
/// raw llvm-sys call for the instructions (LLVMSetAtomicSingleThread).
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

/// The atomic read-modify-write operation of an LLVM `atomicrmw`
/// instruction. The two LLVM-19 wrapping increment/decrement operations
/// have no RVSDG representation yet; clang only emits them for the
/// corresponding builtins, which nothing we compile uses.
fn convert_atomic_read_modify_write_op(
    op: llvm_ir::instruction::RMWBinOp,
) -> color_eyre::Result<AtomicRMWOp> {
    use llvm_ir::instruction::RMWBinOp;
    Ok(match op {
        RMWBinOp::Xchg => AtomicRMWOp::Exchange,
        RMWBinOp::Add => AtomicRMWOp::Add,
        RMWBinOp::Sub => AtomicRMWOp::Sub,
        RMWBinOp::And => AtomicRMWOp::And,
        RMWBinOp::Nand => AtomicRMWOp::Nand,
        RMWBinOp::Or => AtomicRMWOp::Or,
        RMWBinOp::Xor => AtomicRMWOp::Xor,
        RMWBinOp::Max => AtomicRMWOp::SignedMax,
        RMWBinOp::Min => AtomicRMWOp::SignedMin,
        RMWBinOp::UMax => AtomicRMWOp::UnsignedMax,
        RMWBinOp::UMin => AtomicRMWOp::UnsignedMin,
        RMWBinOp::FAdd => AtomicRMWOp::FloatAdd,
        RMWBinOp::FSub => AtomicRMWOp::FloatSub,
        RMWBinOp::FMax => AtomicRMWOp::FloatMax,
        RMWBinOp::FMin => AtomicRMWOp::FloatMin,
        other => {
            return Err(color_eyre::eyre::eyre!(
                "atomic read-modify-write operation {other:?} is not supported"
            ));
        }
    })
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

/// Three lifetimes, each load-bearing:
///   - `'rb`: borrow of the surrounding `RegionBuilder` (typically the
///     shortest -- only valid for the duration of one lowering call).
///   - `'g`:  the graph the `RegionBuilder` writes into (borrowed from the
///     enclosing `RVSDGMod`).
///   - `'m`:  the LLVM module + derived per-function data inside `FnCtx`.
///
/// `'g` and `'m` look mergeable but aren't: at the construction site in
/// `lower_fn_body`, the `RVSDGMod` is borrowed mutably while `FnCtx` borrows
/// the LLVM module + dom tables shared-immutably. Tying them together
/// over-constrains the borrow tree and forces `'static`.
pub(in crate::llvm_parser) struct RegionLowerer<'rb, 'g, 'm> {
    pub rb: &'rb mut RegionBuilder<'g>,
    /// The scoped symbol table (one frame per region on the emission
    /// stack). Reads resolve through it with capture-on-demand; writes land
    /// in the current frame. Shared by every nesting level of one
    /// function's emission.
    pub scopes: &'rb mut SymbolScopes,
    pub fn_ctx: &'m FnCtx<'m>,
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    /// A lowerer emitting into `rb`'s region, reading and writing symbols
    /// through `scopes` (whose current frame must be that same region).
    pub(in crate::llvm_parser) fn new(
        rb: &'rb mut RegionBuilder<'g>,
        scopes: &'rb mut SymbolScopes,
        fn_ctx: &'m FnCtx<'m>,
    ) -> Self {
        Self { rb, scopes, fn_ctx }
    }

    /// Lower one LLVM instruction, threading state through.
    ///
    /// Pure ops (arithmetic, casts, comparisons, etc.) leave state unchanged
    /// and return `state` directly. Side-effecting ops (load, store, alloca,
    /// fence, call, atomic ops) consume the state and produce a new one.
    /// Phi nodes are skipped -- they're absorbed into region parameters at
    /// region boundaries, not lowered as instructions.
    pub(crate) fn lower_instruction(
        &mut self,
        state: State,
        inst: &Instruction,
    ) -> color_eyre::Result<State> {
        let new_state = match inst {
            // ---- Pure integer binary ops ----------------------------------
            Instruction::Add(i) => {
                self.binary(i, BinaryOp::Add, ArithFlags::wrap(i.nsw, i.nuw))?;
                state
            }
            Instruction::Sub(i) => {
                self.binary(i, BinaryOp::Sub, ArithFlags::wrap(i.nsw, i.nuw))?;
                state
            }
            Instruction::Mul(i) => {
                self.binary(i, BinaryOp::Mul, ArithFlags::wrap(i.nsw, i.nuw))?;
                state
            }
            Instruction::UDiv(i) => {
                self.binary(i, BinaryOp::UnsignedDiv, ArithFlags::exact(i.exact))?;
                state
            }
            Instruction::SDiv(i) => {
                self.binary(i, BinaryOp::SignedDiv, ArithFlags::exact(i.exact))?;
                state
            }
            Instruction::URem(i) => {
                self.binary(i, BinaryOp::UnsignedRem, ArithFlags::default())?;
                state
            }
            Instruction::SRem(i) => {
                self.binary(i, BinaryOp::SignedRem, ArithFlags::default())?;
                state
            }
            Instruction::And(i) => {
                self.binary(i, BinaryOp::And, ArithFlags::default())?;
                state
            }
            Instruction::Or(i) => {
                self.binary(i, BinaryOp::Or, ArithFlags::default())?;
                state
            }
            Instruction::Xor(i) => {
                self.binary(i, BinaryOp::Xor, ArithFlags::default())?;
                state
            }
            Instruction::Shl(i) => {
                self.binary(i, BinaryOp::ShiftLeft, ArithFlags::wrap(i.nsw, i.nuw))?;
                state
            }
            Instruction::LShr(i) => {
                self.binary(i, BinaryOp::LogicalShiftRight, ArithFlags::exact(i.exact))?;
                state
            }
            Instruction::AShr(i) => {
                self.binary(i, BinaryOp::ArithShiftRight, ArithFlags::exact(i.exact))?;
                state
            }

            // ---- Pure float binary ops ------------------------------------
            Instruction::FAdd(i) => {
                self.binary(i, BinaryOp::FloatAdd, ArithFlags::default())?;
                state
            }
            Instruction::FSub(i) => {
                self.binary(i, BinaryOp::FloatSub, ArithFlags::default())?;
                state
            }
            Instruction::FMul(i) => {
                self.binary(i, BinaryOp::FloatMul, ArithFlags::default())?;
                state
            }
            Instruction::FDiv(i) => {
                self.binary(i, BinaryOp::FloatDiv, ArithFlags::default())?;
                state
            }
            Instruction::FRem(i) => {
                self.binary(i, BinaryOp::FloatRem, ArithFlags::default())?;
                state
            }
            Instruction::FNeg(i) => {
                self.unary(i, UnaryOp::FloatNeg)?;
                state
            }

            // ---- Pure vector / aggregate ops ------------------------------
            Instruction::ExtractElement(i) => {
                self.extract_element(i)?;
                state
            }
            Instruction::InsertElement(i) => {
                self.insert_element(i)?;
                state
            }
            Instruction::ShuffleVector(_) => {
                // Mask is a constant vector -- needs per-element decomposition.
                // Skipped until vector-constant lowering and undef-element handling are decided.
                todo!("shufflevector")
            }
            Instruction::ExtractValue(i) => {
                self.extract_value(i)?;
                state
            }
            Instruction::InsertValue(i) => {
                self.insert_value(i)?;
                state
            }

            // ---- Memory ops (state-threading) -----------------------------
            Instruction::Alloca(i) => self.alloca(state, i)?.state,
            Instruction::Load(i) => self.load(state, i)?.state,
            Instruction::Store(i) => self.store(state, i)?,
            Instruction::Fence(i) => self.fence(state, i),
            Instruction::CmpXchg(i) => self.compare_and_swap(state, i)?,
            Instruction::AtomicRMW(i) => self.atomic_read_modify_write(state, i)?,
            Instruction::GetElementPtr(i) => {
                self.get_element_ptr(i)?;
                state
            }

            // ---- Pure casts ------------------------------------------------
            Instruction::Trunc(i) => {
                self.cast(i, CastOp::Truncate)?;
                state
            }
            Instruction::ZExt(i) => {
                self.cast(i, CastOp::ZeroExtend)?;
                state
            }
            Instruction::SExt(i) => {
                self.cast(i, CastOp::SignExtend)?;
                state
            }
            Instruction::FPTrunc(i) => {
                self.cast(i, CastOp::FloatTruncate)?;
                state
            }
            Instruction::FPExt(i) => {
                self.cast(i, CastOp::FloatExtend)?;
                state
            }
            Instruction::FPToUI(i) => {
                self.cast(i, CastOp::FloatToUnsigned)?;
                state
            }
            Instruction::FPToSI(i) => {
                self.cast(i, CastOp::FloatToSigned)?;
                state
            }
            Instruction::UIToFP(i) => {
                self.cast(i, CastOp::UnsignedToFloat)?;
                state
            }
            Instruction::SIToFP(i) => {
                self.cast(i, CastOp::SignedToFloat)?;
                state
            }
            Instruction::PtrToInt(i) => {
                self.cast(i, CastOp::PtrToInt)?;
                state
            }
            // LLVM 22 instruction (address extraction for fat/tagged
            // pointers); cannot occur in LLVM 19 input, which is the only
            // version this pipeline builds against.
            Instruction::PtrToAddr(_) => {
                return Err(color_eyre::eyre::eyre!(
                    "ptrtoaddr is an LLVM 22+ instruction; not supported"
                ));
            }
            Instruction::IntToPtr(i) => {
                self.cast(i, CastOp::IntToPtr)?;
                state
            }
            Instruction::BitCast(i) => {
                self.cast(i, CastOp::Bitcast)?;
                state
            }
            Instruction::AddrSpaceCast(i) => {
                self.cast(i, CastOp::Bitcast)?;
                state
            }

            // ---- Pure comparisons / select / freeze -----------------------
            Instruction::ICmp(i) => {
                self.icmp(i)?;
                state
            }
            Instruction::FCmp(i) => {
                self.fcmp(i)?;
                state
            }
            // Phi nodes are absorbed into region parameters elsewhere; no-op here.
            Instruction::Phi(_) => state,
            Instruction::Select(i) => {
                self.select(i)?;
                state
            }
            Instruction::Freeze(i) => {
                self.freeze(i)?;
                state
            }

            // ---- Call (state-threading) ------------------------------------
            Instruction::Call(i) => self.call(state, i)?,

            // ---- Unmodelled ------------------------------------------------
            Instruction::VAArg(_) => todo!("VAArg"),
            Instruction::LandingPad(_) => todo!("LandingPad"),
            Instruction::CatchPad(_) => todo!("CatchPad"),
            Instruction::CleanupPad(_) => todo!("CleanupPad"),
        };
        Ok(new_state)
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
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let dest = inst.get_result();
        let val = self.rb.binary(op, flags, left, right, ty);
        self.scopes.bind_name(dest, val);
        Ok(val)
    }

    fn unary<I>(&mut self, inst: &I, op: UnaryOp) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::UnaryOp + HasResult + Typed,
    {
        let operand = self.operand(inst.get_operand())?;
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let dest = inst.get_result();
        let val = self.rb.unary(op, operand, ty);
        self.scopes.bind_name(dest, val);
        Ok(val)
    }

    fn cast<I>(&mut self, inst: &I, op: CastOp) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::UnaryOp + HasResult + Typed,
    {
        let operand = self.operand(inst.get_operand())?;
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let dest = inst.get_result();
        let val = self.rb.cast(op, operand, ty);
        self.scopes.bind_name(dest, val);
        Ok(val)
    }

    fn extract_value(
        &mut self,
        inst: &llvm_ir::instruction::ExtractValue,
    ) -> color_eyre::Result<ValueId> {
        let aggregate = self.operand(&inst.aggregate)?;
        // A compare-and-swap's `{old value, success flag}` pair has no
        // aggregate value in the RVSDG; its fields are the node's two
        // projections, which the builder lays out directly after the node.
        if matches!(
            self.rb.graph.values[aggregate.0 as usize].kind,
            crate::rvsdg::ValueKind::CompareAndSwap { .. }
        ) {
            let &[index] = inst.indices.as_slice() else {
                return Err(color_eyre::eyre::eyre!(
                    "extractvalue on a compare-and-swap pair takes one index, got {:?}",
                    inst.indices
                ));
            };
            if index > 1 {
                return Err(color_eyre::eyre::eyre!(
                    "extractvalue index {index} out of range for a compare-and-swap pair"
                ));
            }
            let projection = self.rb.graph.projection_of(aggregate, index as u16);
            self.scopes.bind_name(&inst.dest, projection);
            return Ok(projection);
        }
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let field_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let val = self.rb.extract_field(aggregate, &inst.indices, field_type);
        self.scopes.bind_name(&inst.dest, val);
        Ok(val)
    }

    fn insert_value(
        &mut self,
        inst: &llvm_ir::instruction::InsertValue,
    ) -> color_eyre::Result<ValueId> {
        let aggregate = self.operand(&inst.aggregate)?;
        let element = self.operand(&inst.element)?;
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let aggregate_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let val = self
            .rb
            .insert_field(aggregate, element, &inst.indices, aggregate_type);
        self.scopes.bind_name(&inst.dest, val);
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
            .convert_type_ref(&inst.source_element_type, self.fn_ctx.llvm_mod)?;
        let indices = inst
            .indices
            .iter()
            .map(|op| self.operand(op))
            .collect::<Result<Vec<_>, _>>()?;
        let llvm_result_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let result_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_result_ty, self.fn_ctx.llvm_mod)?;
        let val = self
            .rb
            .ptr_offset(base, base_type, &indices, result_type, inst.in_bounds);
        self.scopes.bind_name(&inst.dest, val);
        Ok(val)
    }

    fn icmp(&mut self, inst: &llvm_ir::instruction::ICmp) -> color_eyre::Result<ValueId> {
        let left = self.operand(&inst.operand0)?;
        let right = self.operand(&inst.operand1)?;
        let pred = convert_int_pred(inst.predicate);
        let val = self.rb.icmp(pred, left, right);
        self.scopes.bind_name(&inst.dest, val);
        Ok(val)
    }

    fn fcmp(&mut self, inst: &llvm_ir::instruction::FCmp) -> color_eyre::Result<ValueId> {
        let left = self.operand(&inst.operand0)?;
        let right = self.operand(&inst.operand1)?;
        let pred = convert_fp_pred(inst.predicate);
        let val = self.rb.fcmp(pred, left, right);
        self.scopes.bind_name(&inst.dest, val);
        Ok(val)
    }

    fn load(
        &mut self,
        state: State,
        inst: &llvm_ir::instruction::Load,
    ) -> color_eyre::Result<LoadResult> {
        let addr = self.operand(&inst.address)?;
        let loaded_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&inst.loaded_ty, self.fn_ctx.llvm_mod)?;
        let align = (inst.alignment != 0).then_some(inst.alignment);

        let result = match &inst.atomicity {
            Some(at) => self.rb.atomic_load(
                state,
                addr,
                loaded_type,
                convert_mem_ordering(at.mem_ordering),
                align,
                inst.volatile,
            ),
            None => self.rb.load(state, addr, loaded_type, align, inst.volatile),
        };
        self.scopes.bind_name(&inst.dest, result.value);
        Ok(result)
    }

    fn store(
        &mut self,
        state: State,
        inst: &llvm_ir::instruction::Store,
    ) -> color_eyre::Result<State> {
        let addr = self.operand(&inst.address)?;
        let value = self.operand(&inst.value)?;
        let align = (inst.alignment != 0).then_some(inst.alignment);

        Ok(match &inst.atomicity {
            Some(at) => self.rb.atomic_store(
                state,
                addr,
                value,
                convert_mem_ordering(at.mem_ordering),
                align,
                inst.volatile,
            ),
            None => self.rb.store(state, addr, value, align, inst.volatile),
        })
    }

    fn alloca(
        &mut self,
        state: State,
        inst: &llvm_ir::instruction::Alloca,
    ) -> color_eyre::Result<AllocaResult> {
        let count = self.operand(&inst.num_elements)?;
        let elem_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&inst.allocated_type, self.fn_ctx.llvm_mod)?;
        let llvm_ptr_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let ptr_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ptr_ty, self.fn_ctx.llvm_mod)?;

        let result = self.rb.alloca(state, elem_type, count, ptr_type);
        self.scopes.bind_name(&inst.dest, result.ptr);
        Ok(result)
    }

    fn fence(&mut self, state: State, inst: &llvm_ir::instruction::Fence) -> State {
        let ordering = convert_mem_ordering(inst.atomicity.mem_ordering);
        self.rb.fence(state, ordering)
    }

    /// Lower an LLVM `cmpxchg` instruction: an atomic compare-and-swap.
    fn compare_and_swap(
        &mut self,
        state: State,
        inst: &llvm_ir::instruction::CmpXchg,
    ) -> color_eyre::Result<State> {
        let addr = self.operand(&inst.address)?;
        let expected = self.operand(&inst.expected)?;
        let desired = self.operand(&inst.replacement)?;
        let value_type = self.rb.graph.values[expected.0 as usize].ty;
        // `inst.weak` is deliberately dropped: the node is always a strong
        // compare-and-swap, which never fails spuriously and is therefore a
        // valid implementation of the weak form -- ALWAYS CORRECT, but not
        // always ideal. On x86-64 and ARMv8.1+LSE the two compile
        // identically; on pre-LSE AArch64 (load-linked/store-conditional)
        // strong forces a retry loop nested inside the user's own CAS
        // loop. Restoring weak needs a `weak: bool` here and one raw
        // llvm-sys call at lowering (LLVMSetWeak on the cmpxchg
        // instruction; inkwell has no wrapper but its AsValueRef trait is
        // public).
        let result = self.rb.compare_and_swap(
            state,
            addr,
            expected,
            desired,
            convert_mem_ordering(inst.atomicity.mem_ordering),
            convert_mem_ordering(inst.failure_memory_ordering),
            value_type,
            inst.volatile,
        );
        // The instruction produces an `{old value, success flag}` pair
        // consumed by extractvalue; the RVSDG has no aggregate for it -- the
        // fields are the node's two projections. The destination binds the
        // NODE so `extract_value` can recognise the kind and route to the
        // matching projection.
        self.scopes.bind_name(&inst.dest, result.state.0);
        Ok(result.state)
    }

    /// Lower an LLVM `atomicrmw` instruction: an atomic read-modify-write
    /// returning the value the memory held before the operation.
    fn atomic_read_modify_write(
        &mut self,
        state: State,
        inst: &llvm_ir::instruction::AtomicRMW,
    ) -> color_eyre::Result<State> {
        let addr = self.operand(&inst.address)?;
        let value = self.operand(&inst.value)?;
        let value_type = self.rb.graph.values[value.0 as usize].ty;
        let op = convert_atomic_read_modify_write_op(inst.operation)?;
        let result = self.rb.atomic_read_modify_write(
            state,
            addr,
            value,
            op,
            convert_mem_ordering(inst.atomicity.mem_ordering),
            value_type,
            inst.volatile,
        );
        self.scopes.bind_name(&inst.dest, result.value);
        Ok(result.state)
    }

    fn select(&mut self, inst: &llvm_ir::instruction::Select) -> color_eyre::Result<ValueId> {
        let cond = self.operand(&inst.condition)?;
        let t = self.operand(&inst.true_value)?;
        let f = self.operand(&inst.false_value)?;
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let val = self.rb.ternary(cond, t, f, ty);
        self.scopes.bind_name(&inst.dest, val);
        Ok(val)
    }

    fn freeze<I>(&mut self, inst: &I) -> color_eyre::Result<ValueId>
    where
        I: llvm_ir::instruction::UnaryOp + HasResult + Typed,
    {
        let operand = self.operand(inst.get_operand())?;
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let ty = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let dest = inst.get_result();
        let val = self.rb.freeze(operand, ty);
        self.scopes.bind_name(dest, val);
        Ok(val)
    }

    /// Build the control/predicate value for a `switch` and its arm targets:
    /// arm 0 is the default, arms `1..=N` the cases in declaration order. The
    /// `match` maps each case value to its arm index; any other value to 0
    /// (default). Returns the control predicate and the arm-target list.
    pub(in crate::llvm_parser) fn switch_predicate(
        &mut self,
        switch: &llvm_ir::terminator::Switch,
    ) -> color_eyre::Result<(ValueId, Vec<BasicBlockId>)> {
        let operand = self.operand(&switch.operand)?;
        let mut targets = Vec::with_capacity(switch.dests.len() + 1);
        targets.push(*self.fn_ctx.bb_mapper.get_expect(&switch.default_dest));
        let mut arms: Vec<MatchArm> = Vec::with_capacity(switch.dests.len());
        for (case_index, (case_const, dest)) in switch.dests.iter().enumerate() {
            targets.push(*self.fn_ctx.bb_mapper.get_expect(dest));
            // Case values are integer constants; read the value to key the match.
            let value = match case_const.as_ref() {
                llvm_ir::Constant::Int { value, .. } => *value as i64,
                other => {
                    return Err(color_eyre::eyre::eyre!(
                        "switch case value is not an integer: {other:?}"
                    ));
                }
            };
            // Arm 0 is the default, so case `k` is arm `k + 1`.
            arms.push(MatchArm {
                value,
                alternative: case_index as u32 + 1,
            });
        }
        let alternatives = targets.len() as u32;
        let predicate = self.rb.match_op(operand, &arms, 0, alternatives);
        Ok((predicate, targets))
    }

    pub(super) fn operand(&mut self, op: &Operand) -> color_eyre::Result<ValueId> {
        match op {
            Operand::LocalOperand { name, .. } => self
                .scopes
                .resolve_name(self.rb.graph, name)
                .ok_or_else(|| color_eyre::eyre::eyre!("ssa value {name} used before definition")),
            Operand::ConstantOperand(constant_ref) => {
                let const_id = self
                    .rb
                    .graph
                    .convert_const_ref(constant_ref.clone(), self.fn_ctx.llvm_mod)?;
                let ty = self.rb.graph.constants.get(const_id).ty;
                Ok(self.rb.const_pool_ref(const_id, ty))
            }
            Operand::MetadataOperand => {
                todo!("MetadataOperand is currently unsupported within llvm_ir")
            }
        }
    }
}
