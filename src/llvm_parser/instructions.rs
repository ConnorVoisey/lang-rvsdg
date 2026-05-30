use crate::{
    llvm_parser::FnCtx,
    rvsdg::{
        ArithFlags, BinaryOp, CastOp, FCmpPred, ICmpPred, MemoryOrdering, State, UnaryOp, ValueId,
        builder::{AllocaResult, LoadResult, RegionBuilder},
    },
};
use llvm_ir::{
    FPPredicate, Instruction, IntPredicate, Name, Operand,
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

/// Visits every `Operand` referenced by an instruction.
///
/// "Referenced" means: an operand whose value is used as input to the instruction.
/// Instruction destinations (`HasResult::get_result`) are NOT visited; this is for
/// scanning use-sites only, call separately for definitions if needed.
///
/// Coverage matches `RegionLowerer::lower_instruction`. Variants we don't yet
/// lower (`VAArg`, EH pads) are also unmodelled here, we panic with a clear
/// message rather than silently skipping, because missing an operand silently
/// produces wrong RVSDG (an instruction inside a region referencing an outer
/// value that wasn't lifted to a region input).
///
/// Calling convention: `f` is invoked once per operand position. For multi-operand
/// instructions like `Call` and `GetElementPtr`, `f` is invoked in stable order
/// (callee/base first, then arguments/indices in their LLVM order).
///
/// The explicit `'a` lifetime ties the callback's `&'a Operand` parameter to the
/// input instruction's lifetime — without it, the callback is inferred as
/// `for<'a> FnMut(&'a Operand)` (HRTB), which makes the closure unable to store
/// references extracted from operands into outer collections like `HashSet<&Name>`.
pub(super) fn for_each_operand<'a, F: FnMut(&'a Operand)>(inst: &'a Instruction, mut f: F) {
    match inst {
        // Integer binary
        Instruction::Add(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::Sub(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::Mul(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::UDiv(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::SDiv(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::URem(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::SRem(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::And(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::Or(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::Xor(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::Shl(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::LShr(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::AShr(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }

        // Float binary
        Instruction::FAdd(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::FSub(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::FMul(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::FDiv(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::FRem(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }

        // Unary
        Instruction::FNeg(i) => f(&i.operand),
        Instruction::Trunc(i) => f(&i.operand),
        Instruction::ZExt(i) => f(&i.operand),
        Instruction::SExt(i) => f(&i.operand),
        Instruction::FPTrunc(i) => f(&i.operand),
        Instruction::FPExt(i) => f(&i.operand),
        Instruction::FPToUI(i) => f(&i.operand),
        Instruction::FPToSI(i) => f(&i.operand),
        Instruction::UIToFP(i) => f(&i.operand),
        Instruction::SIToFP(i) => f(&i.operand),
        Instruction::PtrToInt(i) => f(&i.operand),
        Instruction::IntToPtr(i) => f(&i.operand),
        Instruction::BitCast(i) => f(&i.operand),
        Instruction::AddrSpaceCast(i) => f(&i.operand),
        Instruction::Freeze(i) => f(&i.operand),

        // Vector ops
        Instruction::ExtractElement(i) => {
            f(&i.vector);
            f(&i.index);
        }
        Instruction::InsertElement(i) => {
            f(&i.vector);
            f(&i.element);
            f(&i.index);
        }
        Instruction::ShuffleVector(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }

        // Aggregate ops
        Instruction::ExtractValue(i) => f(&i.aggregate),
        Instruction::InsertValue(i) => {
            f(&i.aggregate);
            f(&i.element);
        }

        // Memory
        Instruction::Alloca(i) => f(&i.num_elements),
        Instruction::Load(i) => f(&i.address),
        Instruction::Store(i) => {
            f(&i.address);
            f(&i.value);
        }
        Instruction::Fence(_) => {} // no value operands
        Instruction::CmpXchg(i) => {
            f(&i.address);
            f(&i.expected);
            f(&i.replacement);
        }
        Instruction::AtomicRMW(i) => {
            f(&i.address);
            f(&i.value);
        }
        Instruction::GetElementPtr(i) => {
            f(&i.address);
            for idx in &i.indices {
                f(idx);
            }
        }

        // Comparisons
        Instruction::ICmp(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }
        Instruction::FCmp(i) => {
            f(&i.operand0);
            f(&i.operand1);
        }

        // Select
        Instruction::Select(i) => {
            f(&i.condition);
            f(&i.true_value);
            f(&i.false_value);
        }

        // Phi — each [value, predecessor] pair contributes a value operand
        Instruction::Phi(i) => {
            for (op, _pred) in &i.incoming_values {
                f(op);
            }
        }

        // Call — walk arg operands, plus the callee if it's an indirect call
        Instruction::Call(i) => {
            if let either::Either::Right(callee) = &i.function {
                f(callee);
            }
            for (arg, _attrs) in &i.arguments {
                f(arg);
            }
        }

        // Variadic / EH — not modelled by lower_instruction either.
        // Panic on encounter so the failure surfaces at the operand-walk site,
        // not later as a silent missed-live-in.
        Instruction::VAArg(_) => todo!("for_each_operand: VAArg"),
        Instruction::LandingPad(_) => todo!("for_each_operand: LandingPad"),
        Instruction::CatchPad(_) => todo!("for_each_operand: CatchPad"),
        Instruction::CleanupPad(_) => todo!("for_each_operand: CleanupPad"),
    }
}

/// Returns the SSA destination name of an instruction, if it has one.
///
/// Used by live-in analysis to determine what's "defined inside" a region.
/// Symmetric to `for_each_operand` — that visits uses; this returns the
/// definition. Variants without a value result (`Store`, `Fence`, etc.)
/// return `None`.
pub(super) fn instruction_dest(inst: &Instruction) -> Option<&Name> {
    match inst {
        // Integer binary
        Instruction::Add(i) => Some(i.get_result()),
        Instruction::Sub(i) => Some(i.get_result()),
        Instruction::Mul(i) => Some(i.get_result()),
        Instruction::UDiv(i) => Some(i.get_result()),
        Instruction::SDiv(i) => Some(i.get_result()),
        Instruction::URem(i) => Some(i.get_result()),
        Instruction::SRem(i) => Some(i.get_result()),
        Instruction::And(i) => Some(i.get_result()),
        Instruction::Or(i) => Some(i.get_result()),
        Instruction::Xor(i) => Some(i.get_result()),
        Instruction::Shl(i) => Some(i.get_result()),
        Instruction::LShr(i) => Some(i.get_result()),
        Instruction::AShr(i) => Some(i.get_result()),

        // Float binary
        Instruction::FAdd(i) => Some(i.get_result()),
        Instruction::FSub(i) => Some(i.get_result()),
        Instruction::FMul(i) => Some(i.get_result()),
        Instruction::FDiv(i) => Some(i.get_result()),
        Instruction::FRem(i) => Some(i.get_result()),

        // Unary
        Instruction::FNeg(i) => Some(i.get_result()),
        Instruction::Trunc(i) => Some(i.get_result()),
        Instruction::ZExt(i) => Some(i.get_result()),
        Instruction::SExt(i) => Some(i.get_result()),
        Instruction::FPTrunc(i) => Some(i.get_result()),
        Instruction::FPExt(i) => Some(i.get_result()),
        Instruction::FPToUI(i) => Some(i.get_result()),
        Instruction::FPToSI(i) => Some(i.get_result()),
        Instruction::UIToFP(i) => Some(i.get_result()),
        Instruction::SIToFP(i) => Some(i.get_result()),
        Instruction::PtrToInt(i) => Some(i.get_result()),
        Instruction::IntToPtr(i) => Some(i.get_result()),
        Instruction::BitCast(i) => Some(i.get_result()),
        Instruction::AddrSpaceCast(i) => Some(i.get_result()),
        Instruction::Freeze(i) => Some(i.get_result()),

        // Vector
        Instruction::ExtractElement(i) => Some(i.get_result()),
        Instruction::InsertElement(i) => Some(i.get_result()),
        Instruction::ShuffleVector(i) => Some(i.get_result()),

        // Aggregate
        Instruction::ExtractValue(i) => Some(i.get_result()),
        Instruction::InsertValue(i) => Some(i.get_result()),

        // Memory
        Instruction::Alloca(i) => Some(i.get_result()),
        Instruction::Load(i) => Some(i.get_result()),
        Instruction::Store(_) => None, // void
        Instruction::Fence(_) => None, // void
        Instruction::CmpXchg(i) => Some(i.get_result()),
        Instruction::AtomicRMW(i) => Some(i.get_result()),
        Instruction::GetElementPtr(i) => Some(i.get_result()),

        // Comparisons
        Instruction::ICmp(i) => Some(i.get_result()),
        Instruction::FCmp(i) => Some(i.get_result()),

        // Select / Phi
        Instruction::Select(i) => Some(i.get_result()),
        Instruction::Phi(i) => Some(&i.dest),

        // Call's dest is optional — `None` when the callee returns void.
        Instruction::Call(i) => i.dest.as_ref(),

        // VAArg has a dest in the IR; the rest of EH support is not modelled
        // by lower_instruction. Surface as todo for parity with for_each_operand.
        Instruction::VAArg(_) => todo!("instruction_dest: VAArg"),
        Instruction::LandingPad(_) => todo!("instruction_dest: LandingPad"),
        Instruction::CatchPad(_) => todo!("instruction_dest: CatchPad"),
        Instruction::CleanupPad(_) => todo!("instruction_dest: CleanupPad"),
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

/// Three lifetimes, each load-bearing:
///   - `'rb`: borrow of the surrounding `RegionBuilder` (typically the
///     shortest — only valid for the duration of one lowering call).
///   - `'g`:  the graph the `RegionBuilder` writes into (borrowed from the
///     enclosing `RVSDGMod`).
///   - `'m`:  the LLVM module + derived per-function data inside `FnCtx`.
///
/// `'g` and `'m` look mergeable but aren't: at the construction site in
/// `lower_fn_body`, the `RVSDGMod` is borrowed mutably while `FnCtx` borrows
/// the LLVM module + dom tables shared-immutably. Tying them together
/// over-constrains the borrow tree and forces `'static`.
pub struct RegionLowerer<'rb, 'g, 'm> {
    pub rb: &'rb mut RegionBuilder<'g>,
    pub name_to_value: FxHashMap<Name, ValueId>,
    pub fn_ctx: &'m FnCtx<'m>,
}

impl<'rb, 'g, 'm> RegionLowerer<'rb, 'g, 'm> {
    pub fn new(rb: &'rb mut RegionBuilder<'g>, fn_ctx: &'m FnCtx<'m>) -> Self {
        Self {
            rb,
            fn_ctx,
            name_to_value: FxHashMap::default(),
        }
    }

    /// Lower one LLVM instruction, threading state through.
    ///
    /// Pure ops (arithmetic, casts, comparisons, etc.) leave state unchanged
    /// and return `state` directly. Side-effecting ops (load, store, alloca,
    /// fence, call, atomic ops) consume the state and produce a new one.
    /// Phi nodes are skipped — they're absorbed into region parameters at
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
                // Mask is a constant vector — needs per-element decomposition.
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
            Instruction::CmpXchg(_) => todo!("cmpxchg"),
            Instruction::AtomicRMW(_) => todo!("atomic_rmw"),
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
        let dest = inst.get_result().clone();
        let val = self.rb.binary(op, flags, left, right, ty);
        self.name_to_value.insert(dest, val);
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
        let dest = inst.get_result().clone();
        let val = self.rb.unary(op, operand, ty);
        self.name_to_value.insert(dest, val);
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
        let dest = inst.get_result().clone();
        let val = self.rb.cast(op, operand, ty);
        self.name_to_value.insert(dest, val);
        Ok(val)
    }

    fn extract_value(
        &mut self,
        inst: &llvm_ir::instruction::ExtractValue,
    ) -> color_eyre::Result<ValueId> {
        let aggregate = self.operand(&inst.aggregate)?;
        let llvm_ty = inst.get_type(&self.fn_ctx.llvm_mod.types);
        let field_type = self
            .rb
            .graph
            .types
            .convert_type_ref(&llvm_ty, self.fn_ctx.llvm_mod)?;
        let val = self.rb.extract_field(aggregate, &inst.indices, field_type);
        self.name_to_value.insert(inst.dest.clone(), val);
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
        self.name_to_value.insert(inst.dest.clone(), val);
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
        self.name_to_value.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn icmp(&mut self, inst: &llvm_ir::instruction::ICmp) -> color_eyre::Result<ValueId> {
        let left = self.operand(&inst.operand0)?;
        let right = self.operand(&inst.operand1)?;
        let pred = convert_int_pred(inst.predicate);
        let val = self.rb.icmp(pred, left, right);
        self.name_to_value.insert(inst.dest.clone(), val);
        Ok(val)
    }

    fn fcmp(&mut self, inst: &llvm_ir::instruction::FCmp) -> color_eyre::Result<ValueId> {
        let left = self.operand(&inst.operand0)?;
        let right = self.operand(&inst.operand1)?;
        let pred = convert_fp_pred(inst.predicate);
        let val = self.rb.fcmp(pred, left, right);
        self.name_to_value.insert(inst.dest.clone(), val);
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
            ),
            None => self.rb.load(state, addr, loaded_type, align, inst.volatile),
        };
        self.name_to_value.insert(inst.dest.clone(), result.value);
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
        self.name_to_value.insert(inst.dest.clone(), result.ptr);
        Ok(result)
    }

    fn fence(&mut self, state: State, inst: &llvm_ir::instruction::Fence) -> State {
        let ordering = convert_mem_ordering(inst.atomicity.mem_ordering);
        self.rb.fence(state, ordering)
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
        self.name_to_value.insert(inst.dest.clone(), val);
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
        let dest = inst.get_result().clone();
        let val = self.rb.freeze(operand, ty);
        self.name_to_value.insert(dest, val);
        Ok(val)
    }

    pub(super) fn operand(&mut self, op: &Operand) -> color_eyre::Result<ValueId> {
        match op {
            Operand::LocalOperand { name, .. } => Ok(*self
                .name_to_value
                .get(name)
                .unwrap_or_else(|| panic!("ssa value should already have been defined, {name}"))),
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
