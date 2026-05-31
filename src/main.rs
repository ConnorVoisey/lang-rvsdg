use std::process::ExitCode;

use lang_rvsdg::run_cli;

fn main() -> color_eyre::Result<ExitCode> {
    color_eyre::install()?;
    Ok(ExitCode::from(match run_cli()? {
        Some(v) => v,
        None => 0,
    }))
}
