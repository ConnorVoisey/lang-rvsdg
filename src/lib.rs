use crate::llvm_parser::LLVMParser;
use llvm_ir::Module;
use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

pub mod llvm_parser;
pub mod rvsdg;

fn c_file_to_mod(c_file_path: &Path) -> color_eyre::Result<Module> {
    let bc_output = NamedTempFile::with_suffix(".bc")?;

    let clang_cmd = Command::new("clang-17")
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

    Command::new("opt-17")
        .args(["-passes=mem2reg", "-o", bc_output.path().to_str().unwrap()])
        .stdin(clang_cmd.stdout.unwrap())
        .stdout(Stdio::piped())
        .status()?;
    let module = match Module::from_bc_path(&bc_output.path()) {
        Ok(v) => v,
        Err(e) => panic!("{}", e),
    };

    Ok(module)
}
pub fn compile_c_file(c_file_path: &Path) -> color_eyre::Result<()> {
    let module = c_file_to_mod(c_file_path)?;
    LLVMParser::parser_from_mod(module);
    Ok(())
}
