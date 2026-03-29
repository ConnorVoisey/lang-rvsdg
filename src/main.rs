use std::path::Path;

use lang_rvsdg::compile_c_file;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let c_file_path = Path::new("examples/c/basic.c");
    compile_c_file(c_file_path)?;
    Ok(())
}
