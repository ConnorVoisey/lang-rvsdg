//! Compile-time benchmark harness.
//!
//! Measures each preset's programs across the codegen levels (o0/o2/o3 by
//! default) and writes one durable run record every run. The record is the
//! source of truth the HTML report reads; the terminal table is a glance at
//! the run just taken.
//!
//! The default pass is the deterministic one: each `ours` (program, level)
//! compile is run once under Cachegrind, in parallel across `--threads`.
//! Those counts are machine-independent and reproducible, so they are the
//! regression signal. `--wall` opts into a second, sequential competitive
//! pass -- whole-compile wall + peak RSS, clang as the reference line, and
//! our in-process per-phase breakdown -- which is contention-sensitive and
//! informational on a busy machine. Under `--wall` the Cachegrind pass
//! still runs first and its counts are attached, so valgrind is never run
//! sequentially.
//!
//! Examples:
//!   compile-bench --sqlite ~/code/c/sqlite-amalgamation-3530200
//!   compile-bench --program prog.c -I inc -D FOO --levels o0
//!   compile-bench --wall --polybench ~/code/c/PolyBenchC-4.2.1 --sqlite DIR

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use color_eyre::eyre::{WrapErr, eyre};
use lang_rvsdg::CodegenLevel;
use lang_rvsdg::bench::cachegrind;
use lang_rvsdg::bench::compile_time::{self, Program, RecordOpts};
use lang_rvsdg::bench::counters::{Counters, availability_warning};
use lang_rvsdg::bench::measure::{cpu_governor_warning, median};
use lang_rvsdg::bench::record::{
    Compiler, ConfigRecord, ConfigStatus, MetricSamples, ProgramRecord, RunMeta, RunRecord,
    SCHEMA_VERSION, regenerate_data_js, write_run,
};
use lang_rvsdg::bench::suite::{self, Suite};

/// A whole compile taking longer than this is a finding, not a
/// measurement; the subprocess is killed and the config drops out.
const COMPILE_TIMEOUT: Duration = Duration::from_secs(300);

// Sample counts for the `--wall` competitive pass (the deterministic
// Cachegrind pass is single-shot). Overridable with `--iters`/`--warmup`.
const DEFAULT_ITERS: u32 = 7;
const DEFAULT_WARMUP: u32 = 2;

#[derive(Parser, Debug)]
#[command(name = "compile-bench", about = "RVSDG compile-time benchmark")]
struct Args {
    /// PolyBench 4.2.1 checkout: every kernel it lists becomes a program.
    #[arg(long, value_name = "DIR")]
    polybench: Option<PathBuf>,

    /// sqlite amalgamation directory or `sqlite3.c` file.
    #[arg(long, value_name = "DIR|FILE")]
    sqlite: Option<PathBuf>,

    /// Lua single-file build (`onelua.c`).
    #[arg(long, value_name = "FILE")]
    lua: Option<PathBuf>,

    /// http-server unity build translation unit.
    #[arg(long, value_name = "FILE")]
    unity: Option<PathBuf>,

    /// Ad-hoc single input (.c/.bc/.ll), with `-I`/`-D` below.
    #[arg(long, value_name = "FILE")]
    program: Option<PathBuf>,

    /// Header search path for the `--program` input.
    #[arg(short = 'I', long = "include", value_name = "DIR")]
    include: Vec<String>,

    /// Preprocessor define for the `--program` input.
    #[arg(short = 'D', long = "define", value_name = "NAME[=VALUE]")]
    define: Vec<String>,

    /// Codegen levels to measure, comma-separated (default: o0,o2,o3).
    /// Narrow it (e.g. `--levels o0`) for a faster edit-measure loop.
    #[arg(long, value_delimiter = ',', value_name = "o0[,o2,...]")]
    levels: Option<Vec<CodegenLevel>>,

    /// Skip Cachegrind instruction counting. Cachegrind is the
    /// deterministic, machine-independent signal but runs the whole compile
    /// under a CPU simulator (~10-30x); skipping it leaves only the `--wall`
    /// numbers, so `--no-cachegrind` requires `--wall`.
    #[arg(long, default_value_t = false)]
    no_cachegrind: bool,

    /// Worker threads for the parallel Cachegrind pass (default: CPU
    /// count). Each valgrind instance is memory-heavy and you own the RAM
    /// tradeoff -- lower this for big translation units on a constrained
    /// machine. Applies whenever Cachegrind runs, including the Cachegrind
    /// sub-pass of `--wall`.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,

    /// Also run the sequential competitive pass: native wall + peak RSS
    /// and clang, plus the in-process per-phase breakdown. Slow and noisy
    /// on a busy machine -- for the occasional real-time comparison on a
    /// quiet one. Cachegrind still runs first (in parallel, unless
    /// `--no-cachegrind`) and its counts are attached, so only wall/RSS/
    /// clang are measured sequentially. The default (this off) is the fast
    /// deterministic pass alone.
    #[arg(long, default_value_t = false)]
    wall: bool,

    /// Recorded iterations per config in the `--wall` pass (default 7).
    #[arg(long)]
    iters: Option<u32>,

    /// Discarded warm-up iterations in the `--wall` pass (default 2).
    #[arg(long)]
    warmup: Option<u32>,

    /// Our compiler binary for the end-to-end subprocess measurement
    /// (default: `lang-rvsdg` next to this executable).
    #[arg(long, value_name = "PATH")]
    compiler: Option<PathBuf>,

    /// Directory holding `runs/` and `data.js`.
    #[arg(long, default_value = "bench-results/compile-time")]
    results_dir: PathBuf,
}

/// Default `--threads`: the cores currently *free*, not the total. All
/// logical cores minus the 1-minute load average, so a busy machine is
/// not oversubscribed. At least 1; falls back to the full core count when
/// `/proc/loadavg` is unreadable.
fn default_threads() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let load1 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok())
        .unwrap_or(0.0);
    (cores as f64 - load1).floor().max(1.0) as usize
}

fn gather_suites(args: &Args) -> color_eyre::Result<Vec<Suite>> {
    let mut suites = Vec::new();
    if let Some(dir) = &args.polybench {
        suites.push(suite::polybench(dir)?);
    }
    if let Some(path) = &args.sqlite {
        suites.push(suite::sqlite(path)?);
    }
    if let Some(file) = &args.lua {
        suites.push(suite::lua(file)?);
    }
    if let Some(file) = &args.unity {
        suites.push(suite::unity(file)?);
    }
    if let Some(file) = &args.program {
        suites.push(suite::program(
            file,
            args.include.clone(),
            args.define.clone(),
        )?);
    }
    Ok(suites)
}

/// Locate our compiler binary for the subprocess leg: the `--compiler`
/// override, else `lang-rvsdg` beside this executable (the usual
/// `target/<profile>/` layout).
fn locate_compiler(args: &Args) -> color_eyre::Result<PathBuf> {
    if let Some(path) = &args.compiler {
        return Ok(path.clone());
    }
    let exe = std::env::current_exe().wrap_err("locating this executable")?;
    exe.parent()
        .map(|dir| dir.join("lang-rvsdg"))
        .filter(|p| p.exists())
        .ok_or_else(|| {
            eyre!("lang-rvsdg not found beside this binary; build it or pass --compiler")
        })
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    let suites = gather_suites(&args)?;
    if suites.is_empty() {
        return Err(eyre!(
            "no inputs; pass at least one preset \
             (--polybench/--sqlite/--lua/--unity/--program)"
        ));
    }

    let levels = args
        .levels
        .clone()
        .unwrap_or_else(|| vec![CodegenLevel::O0, CodegenLevel::O2, CodegenLevel::O3]);
    let iters = args.iters.unwrap_or(DEFAULT_ITERS);
    let warmup = args.warmup.unwrap_or(DEFAULT_WARMUP);
    let compiler = locate_compiler(&args)?;

    let mut counters = Counters::new();
    if let Some(warning) = availability_warning(&counters) {
        eprintln!("note: {warning}");
    }
    if let Some(warning) = cpu_governor_warning() {
        eprintln!("note: {warning}");
    }

    // Cachegrind (the deterministic default signal) runs unless opted out
    // or valgrind is missing. It is the only metric the default run
    // records, so if it will not run and `--wall` was not asked for, there
    // is nothing to measure -- fail loudly rather than write empty records.
    let cachegrind = !args.no_cachegrind && cachegrind::available();
    if !args.no_cachegrind && !cachegrind {
        eprintln!("note: valgrind not found; skipping deterministic instruction counting");
    }
    if !cachegrind && !args.wall {
        return Err(eyre!(
            "nothing to measure: Cachegrind is {} and --wall was not set; \
             add --wall for wall/RSS/clang, or {} for the deterministic counts",
            if args.no_cachegrind {
                "disabled (--no-cachegrind)"
            } else {
                "unavailable (valgrind not found)"
            },
            if args.no_cachegrind {
                "drop --no-cachegrind"
            } else {
                "install valgrind"
            },
        ));
    }

    let opts = RecordOpts {
        warmup,
        iters,
        timeout: COMPILE_TIMEOUT,
        cachegrind,
        threads: args.threads,
    };
    let rss_available = Path::new("/usr/bin/time").exists();

    let programs: Vec<&Program> = suites.iter().flat_map(|s| s.programs.iter()).collect();
    // Program names (from file stems) are the key everything downstream
    // correlates on -- the report's grid rows and baseline matching, and
    // the `--wall` pass's merge of the deterministic Cachegrind counts. A
    // collision would silently misattribute data, so require uniqueness.
    let mut seen = std::collections::HashSet::new();
    for program in &programs {
        if !seen.insert(program.name.as_str()) {
            return Err(eyre!(
                "duplicate program name '{}': names come from file stems and must be \
                 unique across all inputs; rename or drop one of the colliding inputs",
                program.name
            ));
        }
    }

    let workers = args.threads.max(1);
    let records = if args.wall {
        // Competitive run: the parallel deterministic Cachegrind pass, then
        // a sequential pass that augments each record with wall + RSS + clang
        // + in-process phases, reusing the already-staged bitcode.
        eprintln!(
            "measuring {} program(s) across {workers} thread(s) (deterministic + wall) ...",
            programs.len(),
        );
        compile_time::measure_competitive_corpus(
            &programs,
            &levels,
            &compiler,
            &opts,
            &mut counters,
        )?
    } else {
        // Default: the parallel deterministic Cachegrind pass only.
        eprintln!(
            "measuring {} program(s) across {workers} thread(s) (deterministic pass) ...",
            programs.len(),
        );
        compile_time::measure_deterministic_corpus(&programs, &levels, &compiler, &opts)?
    };
    if records.is_empty() {
        return Err(eyre!("every program failed to measure"));
    }

    print_table(&records);

    // Every run is durable: the deterministic counts are the point, and the
    // report's baseline is the previous run, so persisting each one is what
    // makes the run-to-run regression view work.
    let meta = RunMeta::capture(iters, warmup, rss_available);
    let record = RunRecord {
        schema_version: SCHEMA_VERSION,
        meta,
        programs: records,
    };
    std::fs::create_dir_all(&args.results_dir)?;
    let run_path = write_run(&args.results_dir, &record).wrap_err("writing run record")?;
    regenerate_data_js(&args.results_dir).wrap_err("regenerating data.js")?;
    eprintln!("run written: {}", run_path.display());
    eprintln!("data.js regenerated in {}", args.results_dir.display());
    Ok(())
}

// -- Terminal table (a glance at the run; the record is authoritative) --

fn ms_cell(samples: &MetricSamples) -> String {
    match median(&samples.wall_ms) {
        Some(ms) => format!("{ms:.1}"),
        None => "-".to_string(),
    }
}

fn rss_cell(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format!("{:.1}MiB", b as f64 / (1024.0 * 1024.0)),
        None => "-".to_string(),
    }
}

fn size_cell(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format!("{:.1}KiB", b as f64 / 1024.0),
        None => "-".to_string(),
    }
}

/// Compact count for the cachegrind sub-line (instr/cycles reach billions,
/// LL misses can be small).
fn human_count(n: u64) -> String {
    let f = n as f64;
    if f >= 1e9 {
        format!("{:.2}G", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1}M", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1}k", f / 1e3)
    } else {
        n.to_string()
    }
}

fn compiler_label(compiler: Compiler) -> &'static str {
    match compiler {
        Compiler::Ours => "ours",
        Compiler::Clang => "clang",
    }
}

fn print_table(records: &[ProgramRecord]) {
    println!(
        "\n{:<26} {:>10} {:>10} {:>10}",
        "program / config", "wall(ms)", "peak-rss", "obj"
    );
    for record in records {
        let tag = if record.verified { "" } else { " [UNVERIFIED]" };
        println!("{} ({} values){}", record.name, record.values, tag);
        if let Some(ir) = &record.emitted_ir {
            println!(
                "   emitted: {} fns, {} blocks, {} instrs, {} phis",
                ir.functions, ir.basic_blocks, ir.instructions, ir.phis
            );
        }
        for config in &record.configs {
            print_config(config);
        }
    }
}

fn print_config(config: &ConfigRecord) {
    let name = format!("{} -{}", compiler_label(config.compiler), config.level);
    // A failed/timed-out compile has no end-to-end numbers; say so
    // instead of printing a row of dashes, and show the stderr tail.
    match config.status {
        ConfigStatus::Failed | ConfigStatus::TimedOut => {
            let tag = if matches!(config.status, ConfigStatus::TimedOut) {
                "TIMEOUT"
            } else {
                "FAILED"
            };
            println!("  {name:<24} {tag}");
            if let Some(error) = &config.error {
                for line in error.lines() {
                    println!("      | {line}");
                }
            }
            return;
        }
        ConfigStatus::Measured => {}
    }
    println!(
        "  {:<24} {:>10} {:>10} {:>10}",
        name,
        ms_cell(&config.end_to_end),
        rss_cell(config.peak_rss_bytes),
        size_cell(config.object_size_bytes),
    );
    // Deterministic Cachegrind counts (ours only; absent when disabled).
    if let (Some(ir), Some(ll), Some(cyc)) = (
        config.cachegrind_ir,
        config.cachegrind_ll_misses,
        config.cachegrind_estimated_cycles,
    ) {
        println!(
            "      cachegrind: {} instr, {} LL-miss, {} est-cyc",
            human_count(ir),
            human_count(ll),
            human_count(cyc),
        );
    }
    // Our per-phase in-process breakdown (clang has none).
    if !config.phases.is_empty() {
        let phases: Vec<String> = config
            .phases
            .iter()
            .map(|p| {
                let ms = median(&p.samples.wall_ms).unwrap_or(0.0);
                format!("{} {ms:.1}", p.phase)
            })
            .collect();
        println!("      phases: {}", phases.join(", "));
        if let Some(misses) = total_cache_misses(config) {
            println!("      cache-misses (in-process, median sum): {misses}");
        }
    }
}

/// Sum of per-phase median cache-miss counts for one config, when every
/// phase reported the counter -- a coarse locality glance (the report
/// does the honest per-phase view).
fn total_cache_misses(config: &ConfigRecord) -> Option<u64> {
    let mut total = 0u64;
    for phase in &config.phases {
        let misses = phase.samples.cache_misses.as_ref()?;
        total += median(misses)?;
    }
    Some(total)
}
