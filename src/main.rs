use clap::Parser;
use std::process::ExitCode;

use lang_rvsdg::{Cli, run_cli};

/// Real Rust-heap numbers for the end-of-compile memory report; the
/// library stays allocator-agnostic.
#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static ALLOCATOR: lang_rvsdg::stats::heap::CountingAllocator =
    lang_rvsdg::stats::heap::CountingAllocator;

/// Heap profiling: dhat replaces the counting allocator (there can be
/// only one global allocator) and writes per-site attribution to
/// dhat-heap.json when the profiler drops.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn main() -> color_eyre::Result<ExitCode> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    color_eyre::install()?;
    let cli = Cli::parse();
    Ok(ExitCode::from(match run_cli(&cli)? {
        Some(v) => v,
        None => 0,
    }))
}
