use llvm_ir::Module;

use crate::rvsdg::{
    ArithFlags, BinaryOp, RVSDGMod,
    func::{FnLinkageType, FnResult},
    types::I32,
};

#[derive(Debug)]
pub struct LLVMParser {}

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

        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], FnLinkageType::External);
        rvsdg.define_fn(main_fn, |rb, entry_state| {
            let a = rb.const_i32(5);
            let b = rb.const_i32(-5);
            let sum = rb.binary(BinaryOp::Add, ArithFlags::default(), a, b, I32);
            FnResult {
                state: entry_state,
                values: vec![sum],
            }
        });
        rvsdg.output_with_llvm().unwrap();
        rvsdg
    }
}
