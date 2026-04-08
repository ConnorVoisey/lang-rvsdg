#[cfg(test)]
pub mod test_utils {
    use crate::rvsdg::RVSDGMod;
    use inkwell::{
        OptimizationLevel,
        context::Context,
        targets::{InitializationConfig, Target},
    };

    macro_rules! define_jit_runner {
        ($name:ident, $ret:ty) => {
            pub fn $name(rvsdg: &RVSDGMod, fn_name: &str) -> $ret {
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
                        .get_function::<unsafe extern "C" fn() -> $ret>(fn_name)
                        .expect("failed to find function")
                };
                unsafe { func.call() }
            }
        };
    }

    define_jit_runner!(jit_run_i32, i32);
    define_jit_runner!(jit_run_i64, i64);
    define_jit_runner!(jit_run_f32, f32);
    define_jit_runner!(jit_run_f64, f64);
}
