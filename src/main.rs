use clap::Parser;
use std::process::ExitCode;

use lang_rvsdg::{Cli, run_cli};

fn main() -> color_eyre::Result<ExitCode> {
    color_eyre::install()?;
    let cli = Cli::parse();
    Ok(ExitCode::from(match run_cli(&cli)? {
        Some(v) => v,
        None => 0,
    }))
}
