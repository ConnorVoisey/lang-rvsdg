//! Per-function SSA read sites at block granularity: for every local
//! name, the blocks in which it is read. One linear pass, run before
//! emission; the emitter consults it at construct assembly so gamma
//! output slots and theta loop-variable slots are created only for
//! symbols something will actually read (see `emit.rs`), instead of
//! speculatively for every written symbol.
//!
//! The filter is conservative by construction: any read the scan
//! cannot attribute keeps the slot, and a kept-but-dead slot is dead
//! node elimination's job exactly as before. Soundness never depends
//! on this table.
//!
//! Attribution rules:
//! - an ordinary operand is a read in the instruction's own block;
//! - a phi incoming is a read in its PREDECESSOR block, which is where
//!   the emitter resolves it (arc payloads apply in the source frame),
//!   so the read lands inside the correct arm or body member set;
//! - terminator operands (branch conditions, switch selectors, return
//!   values) are reads in the terminating block.
//!
//! Only numbered locals are indexed: our clang invocation (-O1)
//! discards value names, so pipeline input never carries string-named
//! locals. A string name (possible in hand-fed .ll input) looks up as
//! "unknown", which the filter treats as read -- the conservative
//! polarity -- rather than as never-read.

use llvm_ir::{Instruction, Name, Operand, Terminator, instruction::BinaryOp as _};
use smallvec::SmallVec;

use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockMapper};

type Blocks = SmallVec<[BasicBlockId; 2]>;

pub(in crate::llvm_parser) struct UseSites {
    numbered: Vec<Blocks>,
}

impl UseSites {
    /// The blocks in which `name` is read: `Some` (possibly empty --
    /// written but never read) for indexed names, `None` for names the
    /// table does not track. Callers must treat `None` as "read
    /// somewhere unknown", never as never-read.
    pub(in crate::llvm_parser) fn read_blocks(&self, name: &Name) -> Option<&[BasicBlockId]> {
        match name {
            Name::Number(n) => Some(self.numbered.get(*n).map_or(&[], |blocks| blocks)),
            Name::Name(_) => None,
        }
    }

    fn record(&mut self, operand: &Operand, block: BasicBlockId) {
        let Operand::LocalOperand {
            name: Name::Number(n),
            ..
        } = operand
        else {
            return;
        };
        if self.numbered.len() <= *n {
            self.numbered.resize_with(n + 1, Blocks::new);
        }
        let blocks = &mut self.numbered[*n];
        // Reads cluster: consecutive operands of the same block dedupe
        // cheaply; cross-block duplicates are harmless (any-outside
        // scans tolerate repeats).
        if blocks.last() != Some(&block) {
            blocks.push(block);
        }
    }

    fn binary<I: llvm_ir::instruction::BinaryOp>(&mut self, inst: &I, block: BasicBlockId) {
        self.record(inst.get_operand0(), block);
        self.record(inst.get_operand1(), block);
    }

    fn unary<I: llvm_ir::instruction::UnaryOp>(&mut self, inst: &I, block: BasicBlockId) {
        self.record(inst.get_operand(), block);
    }

    /// Scan one function. Exhaustive over Instruction and Terminator
    /// (no wildcard arms), the same completeness contract as
    /// `lower_instruction`: a new llvm-ir variant fails compilation
    /// here until its reads are attributed or deliberately ignored.
    pub(in crate::llvm_parser) fn scan(
        func: &llvm_ir::Function,
        bb_mapper: &BasicBlockMapper,
    ) -> Self {
        let mut sites = UseSites {
            numbered: Vec::new(),
        };

        for (index, bb) in func.basic_blocks.iter().enumerate() {
            let here = BasicBlockId(index as u32);
            for inst in &bb.instrs {
                match inst {
                    Instruction::Add(i) => sites.binary(i, here),
                    Instruction::Sub(i) => sites.binary(i, here),
                    Instruction::Mul(i) => sites.binary(i, here),
                    Instruction::UDiv(i) => sites.binary(i, here),
                    Instruction::SDiv(i) => sites.binary(i, here),
                    Instruction::URem(i) => sites.binary(i, here),
                    Instruction::SRem(i) => sites.binary(i, here),
                    Instruction::And(i) => sites.binary(i, here),
                    Instruction::Or(i) => sites.binary(i, here),
                    Instruction::Xor(i) => sites.binary(i, here),
                    Instruction::Shl(i) => sites.binary(i, here),
                    Instruction::LShr(i) => sites.binary(i, here),
                    Instruction::AShr(i) => sites.binary(i, here),
                    Instruction::FAdd(i) => sites.binary(i, here),
                    Instruction::FSub(i) => sites.binary(i, here),
                    Instruction::FMul(i) => sites.binary(i, here),
                    Instruction::FDiv(i) => sites.binary(i, here),
                    Instruction::FRem(i) => sites.binary(i, here),
                    Instruction::FNeg(i) => sites.unary(i, here),
                    Instruction::ExtractElement(i) => {
                        sites.record(&i.vector, here);
                        sites.record(&i.index, here);
                    }
                    Instruction::InsertElement(i) => {
                        sites.record(&i.vector, here);
                        sites.record(&i.element, here);
                        sites.record(&i.index, here);
                    }
                    Instruction::ShuffleVector(i) => {
                        sites.record(&i.operand0, here);
                        sites.record(&i.operand1, here);
                    }
                    Instruction::ExtractValue(i) => sites.record(&i.aggregate, here),
                    Instruction::InsertValue(i) => {
                        sites.record(&i.aggregate, here);
                        sites.record(&i.element, here);
                    }
                    Instruction::Alloca(i) => sites.record(&i.num_elements, here),
                    Instruction::Load(i) => sites.record(&i.address, here),
                    Instruction::Store(i) => {
                        sites.record(&i.address, here);
                        sites.record(&i.value, here);
                    }
                    Instruction::Fence(_) => {}
                    Instruction::CmpXchg(i) => {
                        sites.record(&i.address, here);
                        sites.record(&i.expected, here);
                        sites.record(&i.replacement, here);
                    }
                    Instruction::AtomicRMW(i) => {
                        sites.record(&i.address, here);
                        sites.record(&i.value, here);
                    }
                    Instruction::GetElementPtr(i) => {
                        sites.record(&i.address, here);
                        for index in &i.indices {
                            sites.record(index, here);
                        }
                    }
                    Instruction::Trunc(i) => sites.unary(i, here),
                    Instruction::ZExt(i) => sites.unary(i, here),
                    Instruction::SExt(i) => sites.unary(i, here),
                    Instruction::FPTrunc(i) => sites.unary(i, here),
                    Instruction::FPExt(i) => sites.unary(i, here),
                    Instruction::FPToUI(i) => sites.unary(i, here),
                    Instruction::FPToSI(i) => sites.unary(i, here),
                    Instruction::UIToFP(i) => sites.unary(i, here),
                    Instruction::SIToFP(i) => sites.unary(i, here),
                    Instruction::PtrToInt(i) => sites.unary(i, here),
                    Instruction::PtrToAddr(i) => sites.unary(i, here),
                    Instruction::IntToPtr(i) => sites.unary(i, here),
                    Instruction::BitCast(i) => sites.unary(i, here),
                    Instruction::AddrSpaceCast(i) => sites.unary(i, here),
                    Instruction::ICmp(i) => {
                        sites.record(&i.operand0, here);
                        sites.record(&i.operand1, here);
                    }
                    Instruction::FCmp(i) => {
                        sites.record(&i.operand0, here);
                        sites.record(&i.operand1, here);
                    }
                    Instruction::Phi(i) => {
                        for (operand, pred) in &i.incoming_values {
                            sites.record(operand, *bb_mapper.get_expect(pred));
                        }
                    }
                    Instruction::Select(i) => {
                        sites.record(&i.condition, here);
                        sites.record(&i.true_value, here);
                        sites.record(&i.false_value, here);
                    }
                    Instruction::Freeze(i) => sites.unary(i, here),
                    Instruction::Call(i) => {
                        if let either::Either::Right(callee) = &i.function {
                            sites.record(callee, here);
                        }
                        for (arg, _) in &i.arguments {
                            sites.record(arg, here);
                        }
                    }
                    // Rejected by the instruction lowerer; a compile
                    // that reaches assembly never contained them.
                    Instruction::VAArg(_)
                    | Instruction::LandingPad(_)
                    | Instruction::CatchPad(_)
                    | Instruction::CleanupPad(_) => {}
                }
            }
            match &bb.term {
                Terminator::Ret(ret) => {
                    if let Some(operand) = &ret.return_operand {
                        sites.record(operand, here);
                    }
                }
                Terminator::CondBr(cond_br) => sites.record(&cond_br.condition, here),
                Terminator::Switch(switch) => sites.record(&switch.operand, here),
                Terminator::Br(_) | Terminator::Unreachable(_) => {}
                // Rejected by the parser before emission.
                Terminator::IndirectBr(_)
                | Terminator::Invoke(_)
                | Terminator::Resume(_)
                | Terminator::CleanupRet(_)
                | Terminator::CatchRet(_)
                | Terminator::CatchSwitch(_)
                | Terminator::CallBr(_) => {}
            }
        }

        sites
    }
}
