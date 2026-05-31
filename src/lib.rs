#![warn(missing_debug_implementations)]
// TODO: enable once the API surface is more stable
// #![warn(missing_docs)]
use clap::{ArgGroup, Parser};
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use llvm_ir::Module;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::rvsdg::RVSDGMod;

pub mod llvm_parser;
pub mod rvsdg;

fn c_file_to_mod(c_file_path: &Path) -> color_eyre::Result<Module> {
    // let ll_output = NamedTempFile::with_suffix(".ll")?;
    let ll_output: PathBuf = "c_mod_llvm.ll".into();

    let clang_cmd = Command::new("clang-19")
        .args([
            "-O1",
            "-Xclang",
            "-disable-llvm-passes",
            "-emit-llvm",
            "-c",
            c_file_path.to_str().unwrap_or_default(),
            "-o",
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()?;

    Command::new("opt-19")
        .args([
            "-passes=sroa,mem2reg,loop-simplify,lcssa",
            "-S",
            "-o",
            ll_output.to_str().unwrap(),
        ])
        .stdin(clang_cmd.stdout.unwrap())
        .stdout(Stdio::piped())
        .status()?;

    let llvm_ir_full_text = read_to_string(&ll_output).unwrap();
    println!("Parsed LLVM IR (text): {}", llvm_ir_full_text);

    let module = match Module::from_ir_path(&ll_output) {
        Ok(v) => v,
        Err(e) => panic!("{}", e),
    };

    Ok(module)
}

#[derive(Parser, Debug)]
#[command(name = "RVSDG_CC")]
#[command(version = "0.0")]
#[command(
    about = "Basic c compiler",
    long_about = "Basic c compiler with very little implementation, enough to run some benchmarks to stress the backend"
)]
#[command(group(ArgGroup::new("mode").required(false).args(["output", "run"])))]
pub struct Cli {
    #[arg(short, long)]
    output: Option<String>,

    #[arg(long, short, default_value_t = false)]
    run: bool,

    #[arg(long, default_value_t = false)]
    optimise: bool,

    input: String,
}

pub fn run_cli() -> color_eyre::Result<Option<u8>> {
    let cli = Cli::parse();

    let c_file_path = Path::new(&cli.input);
    let module = c_file_to_mod(c_file_path)?;

    let rvsdg = RVSDGMod::from_llvm_mod(module)?;
    if cli.optimise {
        todo!("run optimisations")
    }

    if cli.run {
        let context = Context::create();
        let module = rvsdg.lower_to_llvm_module(&context)?;
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("failed to create JIT engine");

        let func = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> u8>("main")
                .expect("failed to find main function")
        };
        let res = unsafe { func.call() };
        Ok(Some(res))
    } else {
        let output = match cli.output {
            Some(v) => &v.to_string(),
            None => &rvsdg.mod_name,
        };
        rvsdg.output_with_llvm(output).unwrap();
        Ok(None)
    }
}
