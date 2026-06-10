#![warn(missing_debug_implementations)]
// TODO: enable once the API surface is more stable
// #![warn(missing_docs)]
use clap::{ArgGroup, Parser};
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{InitializationConfig, Target};
use llvm_ir::Module;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use tempfile::NamedTempFile;

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
pub(crate) fn init_llvm_native() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");
    });
}

pub mod llvm_parser;
pub mod rvsdg;

fn c_file_to_mod(
    c_file_path: &Path,
    include_dirs: &[String],
    quiet: bool,
) -> color_eyre::Result<Module> {
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
    clang.arg(c_file_path).args(["-o", "-"]);
    let clang_cmd = clang.stdout(Stdio::piped()).spawn()?;

    // The LLVM opt pass list below contains ONLY structural normalisations
    // that adapt the C source language into a shape this compiler currently
    // ingests. We must not list optimisation passes here (no instcombine,
    // gvn, simplifycfg, etc.): when we later benchmark our own
    // optimisations against LLVM's, the comparison is only meaningful if
    // LLVM has not already done its optimisation work on the input. Every
    // entry below is either a conversion from a C-specific construct into
    // SSA, or a structural normalisation that stands in for a control flow
    // restructuring step we have not yet implemented in our own code.
    //
    // The pass list is being shrunk to `sroa,mem2reg` once construction
    // implements Bahmann, Reissmann, Jahre, Meyer (2015) "Perfect
    // Reconstructability of Control Flow from Demand Dependence Graphs"
    // sections 4.1 and 4.2 directly. See construction_plan.md for the
    // phased plan. Per-pass meaning today:
    //
    //   sroa            : split allocated aggregates into scalars so that
    //                     mem2reg can promote them. C-source conversion;
    //                     stays even after the final pipeline is reached.
    //   mem2reg         : promote alloca plus load and store sequences into
    //                     phi nodes and SSA values. C-source conversion;
    //                     stays after the final pipeline is reached.
    //   loop-simplify   : force every loop into a canonical shape with a
    //                     single preheader, a single back-edge source
    //                     (single latch), and dedicated exit blocks. Stands
    //                     in for our currently absent paper section 4.1
    //                     handling of multi-preheader and multi-latch loops
    //                     via the auxiliary q and r predicates. Dropped in
    //                     Phase 6 once those predicates are implemented.
    //   loop-rotate     : transform every test-first while loop into a
    //                     do-while loop with an outer guard. Stands in for
    //                     our currently absent gating gamma inside theta
    //                     body that the paper section 4.1 produces for
    //                     test-first single-entry loops. Dropped in Phase 1.
    //   lcssa           : insert trivial phi nodes at every loop exit for
    //                     every value defined inside the loop and used
    //                     outside it. Stands in for our currently absent
    //                     demand analysis that the paper's symbolic
    //                     translation performs implicitly. Dropped in
    //                     Phase 5.
    //
    // Two passes are NOT in the list but are widely available and would
    // also be structural normalisations rather than optimisations if added:
    // `unify-loop-exits` (collapses multi-exit loops into single-exit)
    // and `fix-irreducible` (rewrites irreducible cycles into reducible
    // ones via dispatch on an entry predicate). They are deliberately
    // omitted because the paper section 4.1 handling of multi-exit loops
    // via r and irreducible loops via q is what we are implementing
    // directly, in Phases 2 and 3 respectively.
    Command::new("opt-19")
        .args([
            "-passes=sroa,mem2reg,loop-simplify,lcssa",
            "-S",
            "-o",
            ll_output.to_str().unwrap(),
        ])
        .stdin(clang_cmd.stdout.unwrap())
        .stdout(Stdio::piped())
        .status()?;

    let llvm_ir_full_text = read_to_string(&ll_output).unwrap();
    // Diagnostics go to stderr so a compiled program's own stdout (e.g. a
    // csmith checksum) is never mixed with compiler logging.
    if !quiet {
        eprintln!("Parsed LLVM IR (text): {}", llvm_ir_full_text);
    }

    let module = match Module::from_ir_path(&ll_output) {
        Ok(v) => v,
        Err(e) => panic!("{}", e),
    };

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
            input,
        }
    }
}

pub fn run_cli(cli: &Cli) -> color_eyre::Result<Option<u8>> {
    let c_file_path = Path::new(&cli.input);
    let module = c_file_to_mod(c_file_path, &cli.include, cli.quiet)?;

    let rvsdg = RVSDGMod::from_llvm_mod(module)?;
    if cli.optimise {
        todo!("run optimisations")
    }

    if cli.run {
        // Serialised once globally; all per-Context JIT work below runs in parallel.
        init_llvm_native();
        let context = Context::create();
        let module = rvsdg.lower_to_llvm_module(&context)?;
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("failed to create JIT engine");

        let func = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> u8>("main")
                .expect("failed to find main function")
        };
        let res = unsafe { func.call() };
        Ok(Some(res))
    } else {
        let output = match &cli.output {
            Some(v) => &v.to_string(),
            None => &rvsdg.mod_name,
        };
        rvsdg.output_with_llvm(output).unwrap();
        Ok(None)
    }
}
