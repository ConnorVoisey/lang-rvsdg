#![warn(missing_debug_implementations)]
// TODO: enable once the API surface is more stable
// #![warn(missing_docs)]
use llvm_ir::Module;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

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
pub fn compile_c_file(c_file_path: &Path) -> color_eyre::Result<()> {
    let module = c_file_to_mod(c_file_path)?;

    let rvsdg = RVSDGMod::from_llvm_mod(module)?;
    rvsdg.output_with_llvm().unwrap();
    Ok(())
}
