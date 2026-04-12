use lang_rvsdg::rvsdg::{
    ConstValue, GlobalInit, Linkage, RVSDGMod,
    builder::LoopResult,
    func::FnResult,
    types::{ArrayType, I32, I64, PtrType, ScalarType, TypeRef},
};

const BUF_SIZE: usize = 8192;

fn main() -> color_eyre::Result<()> {
    let rvsdg = build_yes();
    rvsdg.output_with_llvm().unwrap();
    Ok(())
}

/// Build the `yes` program: writes "y\n" to stdout in an infinite loop
/// using a large buffer for throughput.
pub fn build_yes() -> RVSDGMod {
    let mut rvsdg = RVSDGMod::new_host(String::from("yes"));

    // Build a BUF_SIZE buffer filled with repeating "y\n"
    let buf: Vec<u8> = b"y\n".iter().copied().cycle().take(BUF_SIZE).collect();
    let arr_id = rvsdg.types.intern_array(ArrayType {
        element: TypeRef::Scalar(ScalarType::I8),
        len: BUF_SIZE as u64,
    });
    let buf_type = TypeRef::Array(arr_id);
    let buf_ptr_type = TypeRef::Ptr(rvsdg.types.intern_ptr(PtrType {
        pointee: Some(buf_type),
        alias_set: None,
        no_escape: false,
    }));
    let buf_const_id = rvsdg.constants.string(buf_type, buf);
    let buf_global = rvsdg.define_global(
        String::from("yes_buf"),
        buf_type,
        GlobalInit::Init(buf_const_id),
        true,
        Linkage::Internal,
    );

    // Declare write(fd: i32, buf: ptr, count: i64) -> i64
    let write_fn = rvsdg.declare_fn(
        String::from("write"),
        &[I32, buf_ptr_type, I64],
        &[I64],
        Linkage::External,
    );

    // main() -> i32
    let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::External);
    let bool_ty = TypeRef::Scalar(ScalarType::Bool);
    rvsdg.define_fn(main_fn, |rb, entry_state| {
        let buf_ptr = rb.global_ref(buf_global, buf_ptr_type);
        let stdout_fd = rb.const_i32(1);
        let buf_len = rb.const_i64(BUF_SIZE as i64);

        // do { write(1, buf, BUF_SIZE) } while(true)
        let res = rb.theta(entry_state, &[], |rb| {
            let call_res = rb.call(write_fn, entry_state, &[stdout_fd, buf_ptr, buf_len]);
            let always_true = rb.constant(bool_ty, ConstValue::Int(1));
            LoopResult {
                condition: always_true,
                next_state: call_res.state,
                next_vars: vec![],
            }
        });

        let zero = rb.const_i32(0);
        FnResult {
            state: res.state,
            values: vec![zero],
        }
    });

    rvsdg
}
