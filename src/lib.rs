#![warn(missing_debug_implementations)]
// TODO: enable once the API surface is more stable
// #![warn(missing_docs)]
use clap::{ArgGroup, Parser};
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{InitializationConfig, Target};
use llvm_ir::Module;
use std::fs::read_to_string;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use tempfile::NamedTempFile;
use tracing_chrome::{ChromeLayerBuilder, FlushGuard};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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

pub mod llvm_parser;
pub mod rvsdg;

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
    tracing_subscriber::registry()
        .with(chrome_layer)
        .try_init()?;
    Ok(guard)
}

#[tracing::instrument(skip_all)]
fn c_file_to_mod(
    c_file_path: &Path,
    include_dirs: &[String],
    defines: &[String],
    quiet: bool,
) -> color_eyre::Result<Module> {
    // A `.ll` input is already LLVM IR: skip the clang + opt frontend entirely
    // and parse it straight through. This is the entry point for a reduced
    // repro -- `llvm-reduce` minimises the `.ll` our pipeline emits, and the
    // minimised file is fed back here directly. A `.c` input runs the normal
    // clang + opt frontend below.
    if c_file_path.extension().and_then(|e| e.to_str()) == Some("ll") {
        let _parse_span = tracing::info_span!("llvm_ir_parse").entered();
        return match Module::from_ir_path(c_file_path) {
            Ok(v) => Ok(v),
            Err(e) => Err(color_eyre::eyre::eyre!("failed to parse LLVM IR: {e}")),
        };
    }

    let ll_file = NamedTempFile::with_suffix(".ll")?;
    let ll_output = ll_file.path();
    // let ll_output: PathBuf = "c_mod_llvm.ll".into();

    let mut clang = Command::new("clang-19");
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

    // Only `mem2reg` runs here: it promotes the allocas clang emits for locals
    // into SSA values and phi nodes, which the construction needs to read the
    // gamma/theta signatures off the phi nodes. Everything else -- restructuring
    // (loop-simplify, loop-rotate, lcssa) and optimisation (sroa, instcombine,
    // gvn, simplifycfg, ...) -- is deliberately omitted: the RVSDG construction
    // restructures raw control flow itself (Bahmann, Reissmann, Jahre, Meyer
    // 2015 sections 4.1/4.2: multi-entry/multi-latch/multi-exit loops and the
    // branch p-demux), and the mid-level optimisation is the RVSDG's job, so
    // letting LLVM do it would both pre-empt that work and bias any later
    // benchmark of our optimisations against LLVM's.
    let ll_output_str = ll_output
        .to_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("temporary .ll path is not valid UTF-8"))?;
    let clang_stdout = clang_cmd.stdout.ok_or_else(|| {
        color_eyre::eyre::eyre!("clang-19 produced no stdout to pipe into opt-19")
    })?;
    {
        let _span = tracing::info_span!("frontend_clang_opt").entered();
        let status = Command::new("opt-19")
            .args(["-passes=mem2reg", "-S", "-o", ll_output_str])
            .stdin(clang_stdout)
            .stdout(Stdio::piped())
            .status()?;
        if !status.success() {
            color_eyre::eyre::bail!("opt-19 failed with status {status}");
        }
    }

    // Diagnostics go to stderr so a compiled program's own stdout (e.g. a
    // csmith checksum) is never mixed with compiler logging.
    if !quiet {
        let llvm_ir_full_text = read_to_string(&ll_output)?;
        eprintln!("Parsed LLVM IR (text): {llvm_ir_full_text}");
    }

    let _parse_span = tracing::info_span!("llvm_ir_parse").entered();
    let module = Module::from_ir_path(&ll_output)
        .map_err(|e| color_eyre::eyre::eyre!("failed to parse LLVM IR: {e}"))?;

    Ok(module)
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
    #[arg(short, long)]
    pub(crate) output: Option<String>,

    #[arg(long, short, default_value_t = false)]
    pub(crate) run: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) optimise: bool,

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
    /// `cc` link step together with the `-I` include paths. Only the primary
    /// `input` goes through the RVSDG pipeline; these are compiled normally.
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

impl Cli {
    pub fn get_run_integration(input: String) -> Self {
        Self {
            output: None,
            run: true,
            optimise: false,
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

    let c_file_path = Path::new(&cli.input);
    let module = c_file_to_mod(c_file_path, &cli.include, &cli.define, cli.quiet)?;

    let rvsdg = RVSDGMod::from_llvm_mod(module)?;
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
    if cli.optimise {
        color_eyre::eyre::bail!("--optimise is not implemented yet");
    }
    if !cli.quiet {
        eprintln!("RVSDG:");
        eprintln!("{rvsdg}");
    }

    if cli.run {
        // Serialised once globally; all per-Context JIT work below runs in parallel.
        init_llvm_native()?;
        let context = Context::create();
        let module = rvsdg.lower_to_llvm_module(&context)?;
        if !cli.quiet {
            eprintln!("LLVM IR:");
            eprintln!("{}", module.print_to_string().to_string());
        }
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| color_eyre::eyre::eyre!("failed to create JIT engine: {e}"))?;

        let func = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> u8>("main")
                .map_err(|e| color_eyre::eyre::eyre!("failed to find `main` to run: {e}"))?
        };
        let res = unsafe { func.call() };
        Ok(Some(res))
    } else {
        let output = match &cli.output {
            Some(v) => &v.to_string(),
            None => &rvsdg.mod_name,
        };
        // Without -o the default output is the module name, which for a
        // .ll input is the input path itself -- a successful link would
        // overwrite the input file with the executable. Refuse instead.
        if Path::new(output).exists()
            && std::fs::canonicalize(output)? == std::fs::canonicalize(c_file_path)?
        {
            color_eyre::eyre::bail!(
                "output path {output} is the input file; pass -o to choose a different output"
            );
        }
        rvsdg.output_with_llvm(output, &cli.link, &cli.link_arg, &cli.include, cli.quiet)?;
        Ok(None)
    }
}
