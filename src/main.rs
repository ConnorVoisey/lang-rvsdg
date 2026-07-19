use clap::Parser;
use std::process::ExitCode;

use lang_rvsdg::{Cli, run_cli};

/// Real Rust-heap numbers for the end-of-compile memory report; the
/// library stays allocator-agnostic.
#[global_allocator]
static ALLOCATOR: lang_rvsdg::stats::heap::CountingAllocator =
    lang_rvsdg::stats::heap::CountingAllocator;

fn main() -> color_eyre::Result<ExitCode> {
    color_eyre::install()?;
    let cli = Cli::parse();
    Ok(ExitCode::from(match run_cli(&cli)? {
        Some(v) => v,
        None => 0,
    }))
}
