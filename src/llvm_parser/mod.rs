use llvm_ir::Module;

use crate::rvsdg::{
    ArithFlags, BinaryOp, GlobalInit, GlobalLinkage, RVSDGMod,
    func::{FnLinkageType, FnResult},
    types::{ArrayType, I32, PtrType, ScalarType, TypeRef},
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

        let c_str = "hello\0";
        let c_str_len = c_str.len();
        let c_str_vec = c_str.as_bytes().to_vec();
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], FnLinkageType::External);
        let arr_id = rvsdg.types.intern_array(ArrayType {
            element: TypeRef::Scalar(ScalarType::I8),
            len: c_str_len as u64,
        });
        let c_str_type = TypeRef::Array(arr_id);
        let c_str_ptr_type = TypeRef::Ptr(rvsdg.types.intern_ptr(PtrType {
            pointee: Some(c_str_type),
            alias_set: None,
            no_escape: false,
        }));
        let puts_fn = rvsdg.declare_fn(
            String::from("puts"),
            &[c_str_ptr_type],
            &[I32],
            FnLinkageType::External,
        );
        let c_str_const_id = rvsdg.constants.string(c_str_type, c_str_vec);
        let str = rvsdg.define_global(
            String::from("string"),
            c_str_type,
            GlobalInit::Init(c_str_const_id),
            true,
            GlobalLinkage::Internal,
        );
        rvsdg.define_fn(main_fn, |rb, entry_state| {
            let str_val = rb.global_ref(str, c_str_ptr_type);
            let puts_res = rb.call(puts_fn, entry_state, &[str_val]);
            let zero = rb.const_i32(0);
            FnResult {
                state: puts_res.state,
                values: vec![zero],
            }
        });
        rvsdg.output_with_llvm().unwrap();
        rvsdg
    }
}
