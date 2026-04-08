#[cfg(test)]
pub mod test_utils {
    use crate::rvsdg::RVSDGMod;
    use inkwell::{
        OptimizationLevel,
        context::Context,
        targets::{InitializationConfig, Target},
    };

    /// Build an RVSDG module, lower to LLVM, JIT-execute the named function,
    /// and return its i32 result.
    pub fn jit_run_i32(rvsdg: &RVSDGMod, fn_name: &str) -> i32 {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");

        let context = Context::create();
        let module = rvsdg
            .lower_to_llvm_module(&context)
            .expect("lowering failed");

        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("failed to create JIT engine");

        let func = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> i32>(fn_name)
                .expect("failed to find function")
        };
        unsafe { func.call() }
    }
}
