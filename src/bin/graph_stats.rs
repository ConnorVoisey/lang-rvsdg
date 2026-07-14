//! Graph census driver: parse inputs through the normal frontend, build
//! the RVSDG, and report shape statistics (see graph_stats_plan.md).
//! Read-only; never emits code.

use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use lang_rvsdg::{c_file_to_mod, rvsdg::RVSDGMod, stats};

#[derive(Parser, Debug)]
#[command(
    name = "graph-stats",
    about = "RVSDG shape census over C/LLVM inputs; design data for optimisation passes"
)]
struct Args {
    /// Inputs (.c, .bc or .ll), each parsed and constructed independently.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Header search path passed to the clang frontend (repeatable).
    #[arg(short = 'I', long = "include", value_name = "DIR")]
    include: Vec<String>,

    /// Preprocessor define passed to the clang frontend (repeatable).
    #[arg(short = 'D', long = "define", value_name = "NAME[=VALUE]")]
    define: Vec<String>,

    /// Write one CSV row per function across all inputs.
    #[arg(long, value_name = "FILE")]
    csv: Option<PathBuf>,

    /// Write one aggregated CSV row per module across all inputs.
    #[arg(long, value_name = "FILE")]
    summary_csv: Option<PathBuf>,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let mut function_csv = args
        .csv
        .as_ref()
        .map(|path| csv::Writer::from_path(path))
        .transpose()?;
    let mut summary_csv = args
        .summary_csv
        .as_ref()
        .map(|path| csv::Writer::from_path(path))
        .transpose()?;

    for input in &args.inputs {
        // Coarse whole-phase timing only, measured HERE so the library
        // and the compiler itself stay probe-free.
        let phase_start = std::time::Instant::now();
        let module = match c_file_to_mod(input, &args.include, &args.define, true) {
            Ok(module) => module,
            Err(error) => {
                writeln!(out, "SKIP {} (frontend: {error:#})", input.display())?;
                continue;
            }
        };
        let frontend_and_parse = phase_start.elapsed();

        let phase_start = std::time::Instant::now();
        let rvsdg = match RVSDGMod::from_llvm_mod(module) {
            Ok(rvsdg) => rvsdg,
            Err(error) => {
                writeln!(out, "SKIP {} (construction: {error:#})", input.display())?;
                continue;
            }
        };
        let construction = phase_start.elapsed();

        // The census only means something over a valid graph.
        let phase_start = std::time::Instant::now();
        let verification_errors = rvsdg.verify();
        let verify = phase_start.elapsed();
        if !verification_errors.is_empty() {
            writeln!(
                out,
                "WARNING {}: {} verification errors; census skipped",
                input.display(),
                verification_errors.len()
            )?;
            continue;
        }

        let phase_start = std::time::Instant::now();
        let mut census = stats::collect(&rvsdg);
        let timing = stats::PhaseTiming {
            frontend_and_parse_ms: frontend_and_parse.as_millis() as u64,
            construction_ms: construction.as_millis() as u64,
            verify_ms: verify.as_millis() as u64,
            census_ms: phase_start.elapsed().as_millis() as u64,
        };
        // The constructed module is named after the frontend's temp
        // file; label census output by the actual input instead.
        census.mod_name = input.display().to_string();
        census.timing = timing;
        census.write_summary(&mut out)?;
        writeln!(out)?;

        if let Some(writer) = function_csv.as_mut() {
            for function in &census.functions {
                writer.serialize(function.row(&census.mod_name))?;
            }
        }
        if let Some(writer) = summary_csv.as_mut() {
            writer.serialize(census.summary_row())?;
        }
    }

    if let Some(mut writer) = function_csv {
        writer.flush()?;
    }
    if let Some(mut writer) = summary_csv {
        writer.flush()?;
    }
    out.flush()?;
    Ok(())
}
