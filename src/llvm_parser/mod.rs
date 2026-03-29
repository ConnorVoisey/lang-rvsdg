use llvm_ir::Module;

use crate::rvsdg::RVSDGMod;

pub struct LLVMParser {
    module: Module,
}

impl LLVMParser {
    pub fn parser_from_mod(module: Module) -> RVSDGMod {
        for func in module.functions {
            for param in func.parameters {
                dbg!(param);
            }
            for bb in func.basic_blocks {
                for inst in bb.instrs {
                    dbg!(inst);
                }
            }
        }
        todo!()
    }
}
