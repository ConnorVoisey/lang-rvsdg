#![warn(missing_debug_implementations)]
// Every unsafe operation needs its own block and SAFETY justification,
// even inside unsafe fns (the GlobalAlloc impl); a bare unsafe fn body
// must not blanket-license its contents.
#![deny(unsafe_op_in_unsafe_fn)]
// TODO: enable once the API surface is more stable
// #![warn(missing_docs)]
use clap::{ArgGroup, Parser};
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{InitializationConfig, Target};
use llvm_ir::Module;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use tempfile::NamedTempFile;
use tracing_chrome::{ChromeLayerBuilder, FlushGuard};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

use crate::rvsdg::RVSDGMod;

/// Initialise LLVM's native target exactly once.
///
/// The target registry is process-global; initialising it mutates global
/// tables and is NOT safe to run concurrently. Once initialised the targets
/// are read-only, so any number of threads can then build their own
/// `Context`, module, and JIT/codegen fully in parallel. The `OnceLock`
/// makes the registry mutation happen a single time regardless of how many
/// threads race into it - this is the only synchronisation parallel LLVM
/// compilation actually needs.
pub(crate) fn init_llvm_native() -> color_eyre::Result<()> {
    // The result of the one-time native-target initialisation is cached so the
    // (process-global, non-reentrant) registry mutation happens exactly once;
    // a failure is reported to every caller rather than panicking the process.
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| Target::initialize_native(&InitializationConfig::default()))
        .as_ref()
        .map(|_| ())
        .map_err(|e| color_eyre::eyre::eyre!("failed to initialize native LLVM target: {e}"))
}

pub mod bench;
pub mod llvm_parser;
pub mod opt;
pub mod rvsdg;
pub mod stats;

/// Install a Chrome-trace `tracing` subscriber writing to `path`, returning a
/// flush guard the caller must hold until the traced work finishes -- dropping
/// it finalises the trace file. The `#[tracing::instrument]` spans throughout
/// construction and lowering are recorded; when this is never called those
/// spans are disabled and effectively free.
///
/// View the output at <https://ui.perfetto.dev> or `chrome://tracing`.
#[must_use]
pub fn init_chrome_tracing(path: &str) -> color_eyre::Result<FlushGuard> {
    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .file(path)
        .include_args(true)
        .build();
    // Spans only: the chrome trace exists for our #[tracing::instrument]
    // spans, not for events. Installed via set_global_default rather than
    // try_init: try_init would also install the log-to-tracing bridge (the
    // tracing-log feature is forced on by other dependencies through
    // feature unification), and the llvm-ir parser logs per-value records
    // at trace level -- the bridge's per-record dynamic dispatch turned a
    // 1-second sqlite3 parse into 19 seconds whenever --trace was active
    // and made the profile lie about where time goes. With no bridge
    // installed, each of those log calls is a single atomic load.
    let subscriber = tracing_subscriber::registry().with(chrome_layer.with_filter(
        tracing_subscriber::filter::filter_fn(|metadata| metadata.is_span()),
    ));
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(guard)
}

/// Parse an input (`.c` through the clang + mem2reg frontend, `.ll`/`.bc`
/// directly) into an llvm-ir module. Public so the fidelity tests can
/// parse the same inputs the compiler does -- one pipeline, not a copy.
#[tracing::instrument(skip_all)]
pub fn c_file_to_mod(
    c_file_path: &Path,
    include_dirs: &[String],
    defines: &[String],
    quiet: bool,
) -> color_eyre::Result<Module> {
    // A `.ll` or `.bc` input is already LLVM IR: skip the clang + opt
    // frontend entirely and parse it straight through. Text `.ll` is the
    // entry point for a reduced repro -- `llvm-reduce` minimises the IR our
    // pipeline emits, and the minimised file is fed back here directly.
    // Bitcode skips LLVM's text lexer, a modest win (sqlite3, 12MB of
    // text: 1.2s -> 1.0s whole-compile). A `.c` input runs the normal
    // clang + opt frontend below.
    match c_file_path.extension().and_then(|e| e.to_str()) {
        Some("ll") => {
            let _parse_span = tracing::info_span!("llvm_ir_parse").entered();
            return Module::from_ir_path(c_file_path)
                .map_err(|e| color_eyre::eyre::eyre!("failed to parse LLVM IR: {e}"));
        }
        Some("bc") => {
            let _parse_span = tracing::info_span!("llvm_ir_parse").entered();
            return Module::from_bc_path(c_file_path)
                .map_err(|e| color_eyre::eyre::eyre!("failed to parse LLVM bitcode: {e}"));
        }
        _ => {}
    }

    let bc_file = c_file_to_bc(c_file_path, include_dirs, defines)?;
    let bc_output = bc_file.path();
    let bc_output_str = bc_output
        .to_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("temporary .bc path is not valid UTF-8"))?;

    // Diagnostics go to stderr so a compiled program's own stdout (e.g. a
    // csmith checksum) is never mixed with compiler logging. The bitcode is
    // disassembled on demand; this path is for humans, not the pipeline.
    if !quiet {
        let dis = Command::new("llvm-dis")
            .args([bc_output_str, "-o", "-"])
            .output()?;
        eprintln!(
            "Parsed LLVM IR (text): {}",
            String::from_utf8_lossy(&dis.stdout)
        );
    }

    let _parse_span = tracing::info_span!("llvm_ir_parse").entered();
    let module = Module::from_bc_path(bc_output)
        .map_err(|e| color_eyre::eyre::eyre!("failed to parse LLVM bitcode: {e}"))?;

    Ok(module)
}

/// Run the scaffold C frontend -- clang generating IR with every LLVM
/// pass disabled, piped into `opt -passes=mem2reg` -- producing a bitcode
/// temp file (deleted on drop). Public so the differential tester can
/// produce ONE .bc per program and time both compilers from the same
/// input: the frontend is shared scaffolding, not part of either side's
/// measured work.
///
/// Only `mem2reg` runs here: it promotes the allocas clang emits for
/// locals into SSA values and phi nodes, which the construction needs to
/// read the gamma/theta signatures off the phi nodes. Everything else --
/// restructuring (loop-simplify, loop-rotate, lcssa) and optimisation
/// (sroa, instcombine, gvn, simplifycfg, ...) -- is deliberately omitted:
/// the RVSDG construction restructures raw control flow itself (Bahmann,
/// Reissmann, Jahre, Meyer 2015 sections 4.1/4.2), and the mid-level
/// optimisation is the RVSDG's job, so letting LLVM do it would both
/// pre-empt that work and bias any later benchmark of our optimisations
/// against LLVM's.
#[tracing::instrument(skip_all)]
pub fn c_file_to_bc(
    c_file_path: &Path,
    include_dirs: &[String],
    defines: &[String],
) -> color_eyre::Result<NamedTempFile> {
    c_file_to_bc_with("clang", "opt", c_file_path, include_dirs, defines)
}

/// [`c_file_to_bc`] with explicit frontend command names, for producing
/// bitcode an OLDER LLVM can read (bitcode is only backwards
/// compatible): the differential tester's jlm comparison feeds one
/// shared clang-17 bitcode to our stack (LLVM 22), clang, and jlc
/// (LLVM 18).
#[tracing::instrument(skip_all)]
pub fn c_file_to_bc_with(
    clang_cmd: &str,
    opt_cmd: &str,
    c_file_path: &Path,
    include_dirs: &[String],
    defines: &[String],
) -> color_eyre::Result<NamedTempFile> {
    let bc_file = NamedTempFile::with_suffix(".bc")?;
    let bc_output_str = bc_file
        .path()
        .to_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("temporary .bc path is not valid UTF-8"))?
        .to_string();

    let mut clang = Command::new(clang_cmd);
    // `-w`: we don't care about clang's warnings on the input here (it's
    // just the SSA frontend), and on fuzzer input they flood stderr and
    // bury our own diagnostics.
    clang.args([
        "-O1",
        "-w",
        "-Xclang",
        "-disable-llvm-passes",
        "-emit-llvm",
        "-c",
    ]);
    // Header search paths (e.g. csmith's runtime header dir). One `-I` per
    // entry so paths with spaces are passed as single arguments.
    for dir in include_dirs {
        clang.arg("-I").arg(dir);
    }
    // Preprocessor defines (e.g. PolyBench's `POLYBENCH_TIME`,
    // `LARGE_DATASET`). Each is passed as `-D<def>`; the value may include an
    // `=` (`NAME=value`).
    for def in defines {
        clang.arg(format!("-D{def}"));
    }
    clang.arg(c_file_path).args(["-o", "-"]);
    let clang_cmd = clang.stdout(Stdio::piped()).spawn()?;

    let clang_stdout = clang_cmd
        .stdout
        .ok_or_else(|| color_eyre::eyre::eyre!("clang produced no stdout to pipe into opt"))?;
    {
        let _span = tracing::info_span!("frontend_clang_opt").entered();
        // Bitcode out (no `-S`): the file exists only to hand the module
        // to the parser, and bitcode skips LLVM's text lexer on the way
        // back in.
        let status = Command::new(opt_cmd)
            .args(["-passes=mem2reg", "-o", &bc_output_str])
            .stdin(clang_stdout)
            .stdout(Stdio::piped())
            .status()?;
        if !status.success() {
            color_eyre::eyre::bail!("opt failed with status {status}");
        }
    }

    Ok(bc_file)
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
    /// Output executable path (an `.o` with the same stem is written next
    /// to it). Defaults to the input file's stem in the current directory.
    #[arg(short, long)]
    pub(crate) output: Option<String>,

    #[arg(long, short, default_value_t = false)]
    pub(crate) run: bool,

    /// Disable the optimisation pipeline (which is on by default), for
    /// A/B measurement of the passes themselves.
    #[arg(long = "no-optimise", default_value_t = false)]
    pub(crate) no_optimise: bool,

    /// Verify the whole graph before any pass runs and again after each
    /// pass, attributing a broken invariant to the pass that broke it.
    /// Available in release builds (debug builds always verify once
    /// after construction).
    #[arg(long = "verify-all", default_value_t = false)]
    pub(crate) verify_all: bool,

    /// Print compile statistics to stderr: graph censuses before and
    /// after the pass pipeline, per-pass reports, and emitted-IR counts.
    #[arg(long, default_value_t = false)]
    pub(crate) stats: bool,

    /// Write the same statistics as JSON to FILE, for machine consumers
    /// (corpus sweeps, regression tracking).
    #[arg(long = "stats-json", value_name = "FILE")]
    pub(crate) stats_json: Option<String>,

    #[arg(long, short, default_value_t = false)]
    pub(crate) quiet: bool,

    /// Header search path passed to the clang frontend (repeatable), e.g.
    /// `-I /usr/include/csmith-2.3.0` so csmith's runtime header resolves.
    #[arg(short = 'I', long = "include", value_name = "DIR")]
    pub(crate) include: Vec<String>,

    /// Preprocessor define passed to the clang frontend (repeatable), e.g.
    /// `-D POLYBENCH_TIME -D LARGE_DATASET`. Accepts `NAME` or `NAME=value`.
    #[arg(short = 'D', long = "define", value_name = "NAME[=VALUE]")]
    pub(crate) define: Vec<String>,

    /// Write a Chrome trace of the compile to this file (view at
    /// <https://ui.perfetto.dev> or `chrome://tracing`). Off when not given.
    #[arg(long, value_name = "FILE")]
    pub(crate) trace: Option<String>,

    /// Extra source/object files to compile and link alongside the compiled
    /// input (repeatable), e.g. `--link utilities/polybench.c` so PolyBench's
    /// harness (`polybench_alloc_data`, timing) resolves. Passed to the final
    /// `cc` link step together with the `-I` include paths and `-D` defines.
    /// Only the primary `input` goes through the RVSDG pipeline; these are
    /// compiled normally.
    #[arg(long = "link", value_name = "FILE")]
    pub(crate) link: Vec<String>,

    /// Extra argument passed verbatim to the final `cc` link step
    /// (repeatable), e.g. `--link-arg -lm` to link against the math
    /// library. Placed after the object files so library flags resolve
    /// the symbols those objects reference.
    #[arg(long = "link-arg", value_name = "ARG", allow_hyphen_values = true)]
    pub(crate) link_arg: Vec<String>,

    pub(crate) input: String,
}

/// Arguments for [`Cli::get_output_integration`], named because input
/// and output are both path strings.
#[derive(Debug)]
pub struct OutputIntegration {
    pub input: String,
    pub output: String,
    pub link: Vec<String>,
}

impl Cli {
    /// Test constructor: compile to an executable, linking extra inputs
    /// (the `--link` flag). Used by the two-translation-unit ABI
    /// fixtures, which need a real linked binary rather than the JIT.
    /// Takes named fields because input and output are both paths -- a
    /// positional swap would compile and overwrite the input fixture.
    pub fn get_output_integration(args: OutputIntegration) -> Self {
        let OutputIntegration {
            input,
            output,
            link,
        } = args;
        Self {
            output: Some(output),
            run: false,
            no_optimise: false,
            // Fixture graphs are tiny; per-pass verification is free
            // coverage that names the guilty pass on failure.
            verify_all: true,
            stats: false,
            stats_json: None,
            quiet: true,
            include: Vec::new(),
            define: Vec::new(),
            trace: None,
            link,
            link_arg: Vec::new(),
            input,
        }
    }

    pub fn get_run_integration(input: String) -> Self {
        Self {
            output: None,
            run: true,
            no_optimise: false,
            verify_all: true,
            stats: false,
            stats_json: None,
            quiet: true,
            include: Vec::new(),
            define: Vec::new(),
            trace: None,
            link: Vec::new(),
            link_arg: Vec::new(),
            input,
        }
    }
}

pub fn run_cli(cli: &Cli) -> color_eyre::Result<Option<u8>> {
    // Held until `run_cli` returns so the trace covers the whole compile; the
    // guard flushes the trace file on drop.
    let _trace_guard = match &cli.trace {
        Some(path) => Some(init_chrome_tracing(path)?),
        None => None,
    };
    // Census collection costs a graph walk per snapshot; everything else
    // in the stats path is timestamps around calls the compile makes
    // anyway.
    let collect_stats = cli.stats || cli.stats_json.is_some();
    if collect_stats {
        // Heap counting costs ~7ns of atomics per alloc/dealloc pair
        // (~0.3-0.5% of a large compile through the llvm-ir parse), so
        // it runs only when the numbers were asked for. The counters
        // absorb the few pre-enable allocations (see stats::heap).
        stats::heap::enable();
    }

    let c_file_path = Path::new(&cli.input);
    let phase_start = std::time::Instant::now();
    let module = c_file_to_mod(c_file_path, &cli.include, &cli.define, cli.quiet)?;
    let frontend_and_parse = phase_start.elapsed();
    let heap_after_parse = stats::heap::live_bytes();

    let phase_start = std::time::Instant::now();
    let mut rvsdg = RVSDGMod::from_llvm_mod(module)?;
    let construction = phase_start.elapsed();

    let phase_start = std::time::Instant::now();
    #[cfg(debug_assertions)]
    {
        let errs = rvsdg.verify();
        if !errs.is_empty() {
            eprintln!("RVSDG:");
            eprintln!("{rvsdg}");
            dbg!(errs);
            panic!("Got errors");
        }
    }
    let verify = phase_start.elapsed();

    // Some iff stats collection was requested: one value whose presence
    // IS the collect decision. Built incrementally through the compile
    // so that whatever exists when a stage fails is still written
    // (write-what-you-have) -- failing inputs are exactly the ones
    // whose shape data has no other source.
    let mut stats_collection: Option<StatsCollection> = if collect_stats {
        let census_start = std::time::Instant::now();
        // The census document holds per-value sample vectors; the human
        // summary is printed and the compact row extracted HERE, so the
        // document dies before the pass pipeline runs and never poisons
        // the heap measurements taken around passes.
        let census_pre_opt = take_census_row(cli, &rvsdg)?;
        Some(StatsCollection {
            phases: stats::PhaseTiming {
                frontend_and_parse_ms: duration_ms(frontend_and_parse),
                construction_ms: duration_ms(construction),
                verify_ms: duration_ms(verify),
                optimise_ms: 0.0,
                census_ms: duration_ms(census_start.elapsed()),
            },
            heap: stats::HeapUsage {
                after_parse_bytes: heap_after_parse,
                live_at_census_bytes: stats::heap::live_bytes(),
                // Stamped when the report is written, so it covers the
                // whole compile.
                peak_bytes: 0,
            },
            census_pre_opt,
            census_post_opt: None,
            emitted_ir: None,
            output_duration: None,
        })
    } else {
        None
    };

    // The JIT context lives OUTSIDE the fallible compile tail so the
    // engine can escape it: stats are then written after compilation
    // but before any guest code runs, so a misbehaving program (exit,
    // abort, non-termination) cannot lose them, and execution time
    // stays outside the output window.
    let jit_context = cli.run.then(Context::create);
    let mut jit_engine = None;
    let mut pipeline = crate::opt::PipelineReport::default();

    let compile_result: color_eyre::Result<()> = (|| {
        if !cli.no_optimise {
            pipeline = rvsdg.optimise_default(cli.verify_all)?;
            // The one place pass reports are printed: for --stats users
            // this lands chronologically between the census summaries.
            if !cli.quiet || cli.stats {
                for report in &pipeline.passes {
                    eprintln!("-- pass -- {report}");
                }
            }
        }
        if let Some(collection) = stats_collection.as_mut() {
            collection.phases.optimise_ms = pipeline
                .passes
                .iter()
                .map(|report| duration_ms(report.duration))
                .sum();
            collection.phases.verify_ms += duration_ms(pipeline.total_verify_duration());
            if !pipeline.passes.is_empty() {
                let census_start = std::time::Instant::now();
                collection.census_post_opt = Some(take_census_row(cli, &rvsdg)?);
                collection.phases.census_ms += duration_ms(census_start.elapsed());
            }
        }

        if !cli.quiet {
            eprintln!("RVSDG:");
            eprintln!("{rvsdg}");
        }

        let phase_start = std::time::Instant::now();
        if let Some(context) = &jit_context {
            // Serialised once globally; all per-Context JIT work below
            // runs in parallel.
            init_llvm_native()?;
            let module = rvsdg.lower_to_llvm_module(context)?;
            if let Some(collection) = stats_collection.as_mut() {
                collection.emitted_ir = Some(rvsdg::lower_to_llvm::emitted_ir_stats(&module));
            }
            if !cli.quiet {
                eprintln!("LLVM IR:");
                eprintln!("{}", module.print_to_string().to_string());
            }
            jit_engine = Some(
                module
                    .create_jit_execution_engine(OptimizationLevel::None)
                    .map_err(|e| color_eyre::eyre::eyre!("failed to create JIT engine: {e}"))?,
            );
        } else {
            // Without -o, derive the output from the input file's stem
            // (`foo.c` -> `./foo`), the rustc convention. The module name is
            // NOT a usable default: for .ll input it is the input path itself,
            // and a successful link once overwrote its own input IR.
            let output = match &cli.output {
                Some(v) => v.clone(),
                None => c_file_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        color_eyre::eyre::eyre!(
                            "cannot derive an output name from input path {}; pass -o",
                            cli.input
                        )
                    })?,
            };
            // Defense in depth behind the stem default: never write the
            // executable over the input file, whatever the paths resolve to.
            if Path::new(&output).exists()
                && std::fs::canonicalize(&output)? == std::fs::canonicalize(c_file_path)?
            {
                color_eyre::eyre::bail!(
                    "output path {output} is the input file; pass -o to choose a different output"
                );
            }
            // A --link-arg value with embedded whitespace that is not a file is
            // almost always several linker flags squeezed into one argument
            // (e.g. '-luring -lpq'); cc receives it as a single unknown option
            // and the resulting linker error does not point back here.
            for arg in &cli.link_arg {
                if arg.contains(char::is_whitespace) && !Path::new(arg).exists() {
                    eprintln!(
                        "warning: --link-arg value {arg:?} contains whitespace and is not a file; \
                         linker flags must be passed one per --link-arg"
                    );
                }
            }
            let emitted_ir = match stats_collection.as_mut() {
                Some(collection) => {
                    Some(collection.emitted_ir.get_or_insert_with(Default::default))
                }
                None => None,
            };
            rvsdg.output_with_llvm(
                &output,
                &cli.link,
                &cli.link_arg,
                &cli.include,
                &cli.define,
                cli.quiet,
                emitted_ir,
            )?;
            // Non-zero only when the binary installs the counting allocator
            // (main does); the rest of RSS is LLVM's C++ heap and the
            // clang/opt frontend subprocesses.
            let heap_peak = stats::heap::peak_bytes();
            if !cli.quiet && heap_peak > 0 {
                eprintln!(
                    "peak rust heap: {:.1} MiB",
                    heap_peak as f64 / (1024.0 * 1024.0)
                );
            }
        }
        if let Some(collection) = stats_collection.as_mut() {
            collection.output_duration = Some(phase_start.elapsed());
        }
        Ok(())
    })();

    // Write-what-you-have: emitted before the compile error (if any)
    // propagates and before any guest code runs. A stats-write failure
    // here loses nothing that already happened (no program has run), so
    // it may surface as an error of its own -- but never at the price
    // of the compile's error, which is the more actionable of the two.
    if let Some(collection) = &stats_collection {
        let compile_error = compile_result
            .as_ref()
            .err()
            .map(|error| format!("{error}"));
        let stats_result =
            write_compile_stats(cli, collection, &pipeline.passes, compile_error.as_deref());
        match (&compile_result, stats_result) {
            (Err(_), Err(stats_error)) => {
                eprintln!("warning: failed to write compile stats: {stats_error}");
            }
            (Ok(()), Err(stats_error)) => return Err(stats_error),
            _ => {}
        }
    }
    compile_result?;

    // Execution, after stats are safely on disk.
    match &jit_engine {
        Some(engine) => {
            // SAFETY: the caller asserts the function's signature; the
            // module's `main` is C main, whose int return we read as
            // its low byte -- the same truncation the OS applies to
            // exit codes.
            let func = unsafe {
                engine
                    .get_function::<unsafe extern "C" fn() -> u8>("main")
                    .map_err(|e| color_eyre::eyre::eyre!("failed to find `main` to run: {e}"))?
            };
            // SAFETY: runs the just-JIT-compiled module in-process; any
            // memory error in the compiled program is the program's
            // (that is what the fixture tests exercise -- crash-mode
            // fixtures use the subprocess harness instead).
            let res = unsafe { func.call() };
            Ok(Some(res))
        }
        None => Ok(None),
    }
}

/// Everything stats collection accumulates across a compile; `Some`-ness
/// of the one `Option<StatsCollection>` in `run_cli` IS the collect
/// decision. Fields fill in as stages complete, so a failed stage still
/// leaves everything before it reportable.
struct StatsCollection {
    /// Per-compile facts, stated once (census rows carry graph shape
    /// only). Timing fields accumulate as stages complete.
    phases: stats::PhaseTiming,
    heap: stats::HeapUsage,
    census_pre_opt: stats::ModuleSummaryRow,
    census_post_opt: Option<stats::ModuleSummaryRow>,
    /// Filled by whichever lowering path runs; None when lowering was
    /// never reached.
    emitted_ir: Option<stats::EmittedIrStats>,
    /// Lowering + codegen + link (or JIT engine build); None when that
    /// stage was never reached or did not finish.
    output_duration: Option<std::time::Duration>,
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Run the census, print its human summary if --stats, and return only
/// the compact row: the full document (per-value sample vectors, MBs at
/// scale) must not outlive the snapshot it describes.
fn take_census_row(cli: &Cli, rvsdg: &RVSDGMod) -> color_eyre::Result<stats::ModuleSummaryRow> {
    let census = stats::collect(rvsdg);
    if cli.stats {
        census.write_summary(&mut std::io::stderr().lock())?;
    }
    Ok(census.summary_row())
}

/// Emit the collected statistics: human summaries to stderr (--stats),
/// one JSON document to a file (--stats-json). `compile_error` is the
/// failure the compile is about to report, if any, so a partial
/// document is distinguishable from a complete one.
fn write_compile_stats(
    cli: &Cli,
    collection: &StatsCollection,
    pass_reports: &[crate::opt::PassReport],
    compile_error: Option<&str>,
) -> color_eyre::Result<()> {
    // Peak covers the whole compile, so it is stamped here at emission
    // rather than at collection time.
    let mut heap = collection.heap;
    heap.peak_bytes = stats::heap::peak_bytes();

    if cli.stats {
        use std::io::Write;
        let mut err = std::io::stderr().lock();
        // Census summaries and pass reports were already printed
        // chronologically as they happened; this block is the trailer.
        if let Some(emitted) = &collection.emitted_ir {
            writeln!(
                err,
                "-- emitted llvm ir -- {} functions, {} blocks, {} instructions, {} phis",
                emitted.functions, emitted.basic_blocks, emitted.instructions, emitted.phis,
            )?;
        }
        let phases = &collection.phases;
        writeln!(
            err,
            "-- phases -- frontend+parse {:.1}ms, construction {:.1}ms, verify {:.1}ms, \
             optimise {:.1}ms, census {:.1}ms",
            phases.frontend_and_parse_ms,
            phases.construction_ms,
            phases.verify_ms,
            phases.optimise_ms,
            phases.census_ms,
        )?;
        if let Some(output_duration) = collection.output_duration {
            writeln!(
                err,
                "-- output (lower + codegen + link) -- {:.1}ms",
                duration_ms(output_duration),
            )?;
        }
        if heap.peak_bytes > 0 {
            let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
            writeln!(
                err,
                "-- rust heap -- after parse {:.1}MiB (llvm-ir AST), live at census {:.1}MiB, \
                 process peak {:.1}MiB; LLVM's C++ heap and subprocesses are outside these",
                mib(heap.after_parse_bytes),
                mib(heap.live_at_census_bytes),
                mib(heap.peak_bytes),
            )?;
        }
        if let Some(error) = compile_error {
            writeln!(
                err,
                "-- compile FAILED (stats above are partial) -- {error}"
            )?;
        }
    }
    if let Some(path) = &cli.stats_json {
        let report = stats::CompileReportJson {
            schema_version: stats::COMPILE_REPORT_SCHEMA_VERSION,
            input: &cli.input,
            phases: collection.phases,
            heap,
            census_pre_opt: &collection.census_pre_opt,
            census_post_opt: collection.census_post_opt.as_ref(),
            passes: pass_reports,
            emitted_ir: collection.emitted_ir,
            output_ms: collection.output_duration.map(duration_ms),
            error: compile_error,
        };
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        // Surface the final flush instead of letting Drop swallow it: a
        // truncated document with exit 0 would read as a good snapshot.
        std::io::Write::flush(&mut writer)?;
    }
    Ok(())
}
