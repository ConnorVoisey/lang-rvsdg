//! Compile-time benchmark: two passes fill the durable record.
//!
//! The frontend (clang + opt -passes=mem2reg) is staged ONCE per program to
//! a `.bc` (see [`crate::stage_to_bitcode`]); every compile below builds
//! that bitcode, so the shared frontend is charged to neither compiler and
//! the numbers isolate the IR-to-object work each actually owns.
//!
//! Default (deterministic) pass ([`measure_deterministic_corpus`]): each
//! ours (program, level) compile is run ONCE under Cachegrind, in parallel
//! across `opts.threads`. The counts barely move run-to-run, so they are
//! the regression signal; only they (and object size) are recorded.
//!
//! `--wall` competitive pass ([`measure_competitive_corpus`]): augments the
//! deterministic records in place with whole-compile wall + peak RSS, a
//! clang reference line, and the in-process per-phase breakdown (see
//! [`counters`]). Sequential, because wall / RSS are contention-sensitive
//! -- informational on a busy machine, not a regression signal.
//!
//! Every metric is stored as RAW PER-ITERATION SAMPLES, never a
//! pre-aggregated summary: the report computes medians, error bars, and
//! significance from the samples, which a min/median/max summary could
//! not support. Comparison and significance live in the HTML report over
//! the stored runs, not here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use color_eyre::eyre::{WrapErr, eyre};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::FileType;

use crate::CodegenLevel;
use crate::bench::cachegrind::{self, CachegrindStats};
use crate::bench::counters::{Counters, PhaseMetrics};
use crate::bench::measure::SampleFailure;
use crate::bench::record::{
    Compiler, ConfigRecord, ConfigStatus, MetricSamples, PassRecord, PhaseRecord, ProgramRecord,
};
use crate::bench::subprocess::{self, Measurement, SubprocessSample};
use crate::rvsdg::RVSDGMod;
use crate::rvsdg::lower_to_llvm::emitted_ir_stats;
use crate::stats::EmittedIrStats;

/// Pipeline phases, in execution order. `Parse` is the llvm-ir crate's
/// cost (context, not the thing under test); the rest are ours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Parse,
    Construct,
    Optimise,
    Lower,
    Codegen,
}

impl Phase {
    /// Every phase in execution order, and the single source of phase
    /// ordering: the sample arrays are sized by `ALL.len()` and indexed by
    /// `phase as usize`, and the record is emitted by iterating this, so
    /// adding or reordering a phase here (and its variant) flows through
    /// without touching the readers.
    const ALL: [Phase; 5] = [
        Phase::Parse,
        Phase::Construct,
        Phase::Optimise,
        Phase::Lower,
        Phase::Codegen,
    ];

    fn label(self) -> &'static str {
        match self {
            Phase::Parse => "parse",
            Phase::Construct => "construct",
            Phase::Optimise => "optimise",
            Phase::Lower => "lower",
            Phase::Codegen => "codegen",
        }
    }
}

/// One corpus entry: an input plus the frontend flags it needs.
#[derive(Debug)]
pub struct Program {
    pub name: String,
    pub input: PathBuf,
    pub includes: Vec<String>,
    pub defines: Vec<String>,
}

/// How thoroughly to measure: the per-config iteration counts (wall pass
/// only), the per-compile timeout, whether to run Cachegrind, and how many
/// worker threads the deterministic pass may use.
#[derive(Debug, Clone, Copy)]
pub struct RecordOpts {
    pub warmup: u32,
    pub iters: u32,
    pub timeout: Duration,
    pub cachegrind: bool,
    pub threads: usize,
}

/// Raw per-iteration samples from the in-process pipeline at one codegen
/// level -- the unaggregated form the record path builds on.
struct PhaseRun {
    /// One vector per phase, indexed by `phase as usize` (see [`Phase::ALL`]).
    phase_samples: [Vec<PhaseMetrics>; Phase::ALL.len()],
    per_pass_samples: Vec<Vec<(String, Duration)>>,
    emitted_ir: Option<EmittedIrStats>,
}

/// Run the in-process pipeline (parse -> construct -> optimise -> lower
/// -> codegen at `codegen`) over `warmup` discarded + `iters` recorded
/// iterations, keeping every sample. Warmup matters because the first
/// pass pays cold caches / page faults, which would inflate the spread.
fn measure_phases(
    bc_path: &Path,
    label: &str,
    codegen: OptimizationLevel,
    warmup: u32,
    iters: u32,
    counters: &mut Counters,
) -> color_eyre::Result<PhaseRun> {
    let mut phase_samples: [Vec<PhaseMetrics>; Phase::ALL.len()] = Default::default();
    let mut per_pass_samples: Vec<Vec<(String, Duration)>> = Vec::with_capacity(iters as usize);
    let mut emitted_ir = None;

    for iteration in 0..(warmup + iters) {
        let recorded = iteration >= warmup;
        let (module, parse) = counters.measure(|| crate::ir_file_to_mod(bc_path));
        let module = module.map_err(|e| eyre!("parsing IR for {label}: {e}"))?;

        let (rvsdg, construct) = counters.measure(|| RVSDGMod::from_llvm_mod(module));
        let mut rvsdg = rvsdg.map_err(|e| eyre!("constructing RVSDG for {label}: {e}"))?;

        let (report, optimise) = counters.measure(|| rvsdg.optimise_default(false));
        let report = report.map_err(|e| eyre!("optimising {label}: {e}"))?;

        // Context out of the timed region; the lowered module borrows it
        // and lives through codegen below, dropping before the context.
        let context = Context::create();
        let (module, lower) = counters.measure(|| rvsdg.lower_to_llvm_module(&context));
        let module = module.map_err(|e| eyre!("lowering {label}: {e}"))?;
        if emitted_ir.is_none() {
            emitted_ir = Some(emitted_ir_stats(&module));
        }

        // Target machine construction is setup, kept out of the timed
        // codegen region; the object goes to a temp file we discard.
        let machine = rvsdg.target_machine(codegen)?;
        let object = tempfile::NamedTempFile::with_suffix(".o")?;
        let (written, codegen_metrics) =
            counters.measure(|| machine.write_to_file(&module, FileType::Object, object.path()));
        written.map_err(|e| eyre!("codegen for {label}: {e}"))?;

        if !recorded {
            continue;
        }
        phase_samples[Phase::Parse as usize].push(parse);
        phase_samples[Phase::Construct as usize].push(construct);
        phase_samples[Phase::Optimise as usize].push(optimise);
        phase_samples[Phase::Lower as usize].push(lower);
        phase_samples[Phase::Codegen as usize].push(codegen_metrics);
        per_pass_samples.push(
            report
                .passes
                .iter()
                .map(|p| (p.pass.to_string(), p.duration))
                .collect(),
        );
    }

    Ok(PhaseRun {
        phase_samples,
        per_pass_samples,
        emitted_ir,
    })
}

// -- Config-matrix orchestration (the durable record) ------------------

fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// Per-iteration in-process samples of one phase as the record's sample
/// vectors. Counter vectors are all-or-nothing: `None` unless every
/// iteration reported the counter.
fn phase_metric_samples(metrics: &[PhaseMetrics]) -> MetricSamples {
    let counter = |pick: fn(&PhaseMetrics) -> Option<u64>| {
        metrics.iter().map(pick).collect::<Option<Vec<u64>>>()
    };
    MetricSamples {
        wall_ms: metrics
            .iter()
            .map(|m| m.wall.as_secs_f64() * 1000.0)
            .collect(),
        cycles: counter(|m| m.cycles),
        instructions: counter(|m| m.instructions),
        cache_misses: counter(|m| m.cache_misses),
        cache_references: counter(|m| m.cache_references),
        allocations: counter(|m| m.allocations),
        alloc_bytes: counter(|m| m.alloc_bytes),
    }
}

/// Subprocess wall samples as a `MetricSamples`; counters are `None`
/// (an opaque subprocess has no per-phase hardware counters).
fn subprocess_wall_samples(samples: &[SubprocessSample]) -> MetricSamples {
    MetricSamples {
        wall_ms: samples
            .iter()
            .map(|s| s.wall.as_secs_f64() * 1000.0)
            .collect(),
        ..MetricSamples::default()
    }
}

/// Convert an in-process run into the record's per-phase / per-pass form.
fn phase_records(run: &PhaseRun) -> Vec<PhaseRecord> {
    Phase::ALL
        .iter()
        .map(|phase| PhaseRecord {
            phase: phase.label().to_string(),
            samples: phase_metric_samples(&run.phase_samples[*phase as usize]),
        })
        .collect()
}

fn pass_records(run: &PhaseRun) -> Vec<PassRecord> {
    let Some(first) = run.per_pass_samples.first() else {
        return Vec::new();
    };
    // The pass list is deterministic across iterations (same staged module,
    // same pipeline), so names come from the first iteration. `get(i)` on
    // the rest keeps a divergent iteration from panicking rather than
    // trusting that invariant with an unchecked index.
    (0..first.len())
        .map(|i| PassRecord {
            name: first[i].0.clone(),
            wall_ms: run
                .per_pass_samples
                .iter()
                .filter_map(|iter| iter.get(i).map(|(_, d)| d.as_secs_f64() * 1000.0))
                .collect(),
        })
        .collect()
}

/// The `ours` driver invocation for one config: compile-only (`-c`) from
/// the staged bitcode at the given backend level. One source of truth for
/// how we invoke our own compiler, shared by both passes.
fn ours_compile_args(level: CodegenLevel, obj: &Path, bc: &Path) -> Vec<String> {
    vec![
        "-q".to_string(),
        "-c".to_string(),
        "--codegen-opt".to_string(),
        level.flag().to_string(),
        "-o".to_string(),
        obj.to_string_lossy().into_owned(),
        bc.to_string_lossy().into_owned(),
    ]
}

/// The clang reference invocation for one config: compile-only from the
/// same staged bitcode at the matching `-O` level.
fn clang_compile_args(level: CodegenLevel, obj: &Path, bc: &Path) -> Vec<String> {
    vec![
        level.clang_flag().to_string(),
        "-w".to_string(),
        "-c".to_string(),
        bc.to_string_lossy().into_owned(),
        "-o".to_string(),
        obj.to_string_lossy().into_owned(),
    ]
}

/// The end-to-end (subprocess) side of one config: the compile outcome
/// plus the two figures that are only meaningful when it succeeded.
struct EndToEnd {
    measurement: Measurement,
    /// Read only on success, so a failed config never inherits a stale
    /// object left by a prior config in the reused temp dir.
    object_size_bytes: Option<u64>,
}

/// Cachegrind is ~10-30x native, so it gets a much longer ceiling than a
/// native compile -- still a ceiling, not an expectation.
const CACHEGRIND_TIMEOUT_MULTIPLIER: u32 = 30;

fn measure_end_to_end(
    program: &str,
    args: &[String],
    opts: &RecordOpts,
    obj_path: &Path,
    io_dir: &Path,
    on_iter: &mut dyn FnMut(u32, u32),
) -> color_eyre::Result<EndToEnd> {
    let measurement = subprocess::measure(
        program,
        args,
        opts.warmup,
        opts.iters,
        opts.timeout,
        io_dir,
        on_iter,
    )?;
    let measured = matches!(measurement, Measurement::Measured(_));
    let object_size_bytes = measured.then(|| file_size(obj_path)).flatten();
    // Wall/RSS/object only. Cachegrind is never run here: the ~30x-slow
    // simulation is the parallel deterministic pass's job, and this compile's
    // config already carries those counts (see `merge_wall`).
    Ok(EndToEnd {
        measurement,
        object_size_bytes,
    })
}

/// Last `STDERR_TAIL_LINES` of a failing compile's stderr, so a broken
/// compile is diagnosable from the record without storing an unbounded
/// log. The tail (not the head) is kept: the fatal message is at the end
/// for our driver's error and near it for clang.
const STDERR_TAIL_LINES: usize = 40;

fn stderr_tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().collect();
    if lines.len() <= STDERR_TAIL_LINES {
        return stderr.trim_end().to_string();
    }
    let tail = lines[lines.len() - STDERR_TAIL_LINES..].join("\n");
    format!("... (truncated, last {STDERR_TAIL_LINES} lines)\n{tail}")
}

/// The measured figures of one config, each optional because a metric is
/// only present when the pass that owns it collected it (the deterministic
/// pass has no wall/RSS; only ours gets Cachegrind).
struct MeasuredConfig {
    end_to_end: MetricSamples,
    peak_rss_bytes: Option<u64>,
    object_size_bytes: Option<u64>,
    cachegrind: Option<CachegrindStats>,
}

/// A measured config record. The single place the Cachegrind stats map
/// onto the record's fields, so a new count is added here once and both
/// passes get it.
fn measured_config(
    compiler: Compiler,
    level: CodegenLevel,
    measured: MeasuredConfig,
    phases: Vec<PhaseRecord>,
    passes: Vec<PassRecord>,
) -> ConfigRecord {
    let MeasuredConfig {
        end_to_end,
        peak_rss_bytes,
        object_size_bytes,
        cachegrind,
    } = measured;
    ConfigRecord {
        compiler,
        level: level.flag().to_string(),
        status: ConfigStatus::Measured,
        error: None,
        end_to_end,
        peak_rss_bytes,
        object_size_bytes,
        cachegrind_ir: cachegrind.map(|c| c.ir),
        cachegrind_ll_misses: cachegrind.map(|c| c.ll_misses),
        cachegrind_total_accesses: cachegrind.map(|c| c.total_accesses),
        cachegrind_estimated_cycles: cachegrind.map(|c| c.estimated_cycles),
        phases,
        passes,
    }
}

/// A failed/timed-out config record. Every measurement-derived field is
/// forced empty here, so a failure can never surface a stale object or a
/// spurious count whatever the caller has to hand.
fn failed_config(
    compiler: Compiler,
    level: CodegenLevel,
    failure: &SampleFailure,
    phases: Vec<PhaseRecord>,
    passes: Vec<PassRecord>,
) -> ConfigRecord {
    ConfigRecord {
        compiler,
        level: level.flag().to_string(),
        status: if failure.timed_out {
            ConfigStatus::TimedOut
        } else {
            ConfigStatus::Failed
        },
        error: Some(stderr_tail(&failure.stderr)),
        end_to_end: MetricSamples::default(),
        peak_rss_bytes: None,
        object_size_bytes: None,
        cachegrind_ir: None,
        cachegrind_ll_misses: None,
        cachegrind_total_accesses: None,
        cachegrind_estimated_cycles: None,
        phases,
        passes,
    }
}

/// A config record from an end-to-end (subprocess) measurement. Used for
/// clang, which carries no Cachegrind counts (only ours is simulated).
fn config_record(
    compiler: Compiler,
    level: CodegenLevel,
    end_to_end: EndToEnd,
    phases: Vec<PhaseRecord>,
    passes: Vec<PassRecord>,
) -> ConfigRecord {
    let EndToEnd {
        measurement,
        object_size_bytes,
    } = end_to_end;
    match measurement {
        Measurement::Measured(samples) => measured_config(
            compiler,
            level,
            MeasuredConfig {
                end_to_end: subprocess_wall_samples(&samples),
                peak_rss_bytes: subprocess::median_peak_rss(&samples),
                object_size_bytes,
                cachegrind: None,
            },
            phases,
            passes,
        ),
        Measurement::Failed(failure) => failed_config(compiler, level, &failure, phases, passes),
    }
}

// -- Deterministic pass (the parallel default) -------------------------
//
// The default full run: for each program, an in-process probe (graph size
// + verify), then every (program, level) ours compile measured under
// Cachegrind. The Cachegrind runs are contention-independent, so they run
// in parallel across `opts.threads`; only the counts are recorded (wall
// and clang are the sequential `--wall` competitive pass, not this one).

/// A program that probed successfully, with a scratch dir the parallel
/// pass writes its objects and Cachegrind output into.
struct Probed {
    name: String,
    // The shared frontend staged once; the ours compile builds this bitcode.
    staged: crate::StagedInput,
    values: usize,
    verified: bool,
    io: tempfile::TempDir,
}

/// Stage the input to IR once and construct the graph in-process, for the
/// deterministic metadata (graph size, and whether `verify()` passes). The
/// staged bitcode is kept so the parallel pass compiles it directly.
fn probe_program(program: &Program) -> color_eyre::Result<Probed> {
    let staged = crate::stage_to_bitcode(&program.input, &program.includes, &program.defines)
        .wrap_err_with(|| format!("frontend failed for {}", program.name))?;
    let rvsdg = RVSDGMod::from_llvm_mod(crate::ir_file_to_mod(staged.path())?)?;
    Ok(Probed {
        name: program.name.clone(),
        values: rvsdg.graphs.iter().fold(0, |acc, g| {
            acc + match g {
                Some(g) => g.value_kinds.len(),
                None => 0,
            }
        }),
        verified: rvsdg.verify().is_empty(),
        staged,
        io: tempfile::tempdir()?,
    })
}

/// The deterministic record for one ours config: a single native compile
/// establishes success + object size (its wall is discarded -- this pass
/// does not record wall), then Cachegrind supplies the deterministic
/// counts. Errors become a failed config, never a fabricated number.
fn deterministic_config(
    prog: &Probed,
    level: CodegenLevel,
    ours_bin: &str,
    opts: &RecordOpts,
) -> ConfigRecord {
    let obj = prog.io.path().join(format!("ours.{}.o", level.flag()));
    // Compile the pre-staged bitcode, so the shared clang + mem2reg frontend
    // is excluded (matching the `--wall` pass and keeping the frontend out
    // of the Cachegrind counts).
    let args = ours_compile_args(level, &obj, prog.staged.path());

    // Single-shot; the parallel pass tracks progress per worker, not per
    // iteration, so the callback is a no-op here.
    let native = subprocess::measure(
        ours_bin,
        &args,
        0,
        1,
        opts.timeout,
        prog.io.path(),
        &mut |_, _| {},
    )
    .unwrap_or_else(|e| {
        Measurement::Failed(SampleFailure {
            timed_out: false,
            stderr: format!("spawn failed: {e}"),
        })
    });

    match native {
        Measurement::Failed(failure) => {
            failed_config(Compiler::Ours, level, &failure, Vec::new(), Vec::new())
        }
        Measurement::Measured(_) => {
            // Read the object size from the clean native compile NOW, before
            // the Cachegrind rerun recompiles to the same path: a Cachegrind
            // timeout would otherwise leave a truncated object and record a
            // bogus size on a config that still counts as measured.
            let object_size_bytes = file_size(&obj);
            let cg = opts
                .cachegrind
                .then(|| {
                    let cg_timeout = opts.timeout.saturating_mul(CACHEGRIND_TIMEOUT_MULTIPLIER);
                    cachegrind::measure(ours_bin, &args, cg_timeout, prog.io.path())
                })
                .flatten();
            // No wall in the deterministic pass -- it is contention-sensitive
            // and lives in the `--wall` competitive pass.
            measured_config(
                Compiler::Ours,
                level,
                MeasuredConfig {
                    end_to_end: MetricSamples::default(),
                    peak_rss_bytes: None,
                    object_size_bytes,
                    cachegrind: cg,
                },
                Vec::new(),
                Vec::new(),
            )
        }
    }
}

/// Probe every program: stage the shared bitcode once and construct the
/// graph in-process for its size and `verify()` result. Sequential so LLVM
/// stays single-threaded; a program that fails to stage or parse is logged
/// and skipped. Each `Probed` keeps its staged bitcode alive for the
/// compile passes to reuse.
fn probe_all(programs: &[&Program]) -> Vec<Probed> {
    let mut probed = Vec::new();
    for program in programs {
        match probe_program(program) {
            Ok(p) => probed.push(p),
            Err(error) => eprintln!("SKIP {} ({error:#})", program.name),
        }
    }
    probed
}

/// The deterministic pass over already-probed programs: every
/// (program, level) ours compile run once under Cachegrind, in parallel
/// across `opts.threads` workers via a largest-first pull queue. Records
/// only the machine-independent counts and object size; wall and clang are
/// the `--wall` pass.
fn deterministic_records(
    probed: &[Probed],
    levels: &[CodegenLevel],
    ours_bin: &str,
    opts: &RecordOpts,
) -> Vec<ProgramRecord> {
    let mut jobs: Vec<(usize, usize)> = (0..probed.len())
        .flat_map(|p| (0..levels.len()).map(move |l| (p, l)))
        .collect();
    // Largest program first (by graph size, a proxy for compile work).
    // Workers pull from the front of this list, so the slowest jobs start
    // early and do not become a lone tail. Stable sort keeps each program's
    // levels in order; results are reordered on assembly, so scheduling
    // order does not affect the record.
    jobs.sort_by_key(|&(p, _)| std::cmp::Reverse(probed[p].values));
    let n_workers = opts.threads.max(1);
    // Progress: an overall bar plus one spinner per worker showing what it
    // is currently compiling, so the largest-first schedule is visible.
    // indicatif's stderr target is TTY-aware, so a redirected log is quiet.
    let bars = MultiProgress::new();
    let overall = bars.add(ProgressBar::new(jobs.len() as u64));
    overall.set_style(
        ProgressStyle::with_template(
            "  cachegrind [{bar:28}] {pos}/{len} ({percent}%, {eta} left)",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    let worker_style = ProgressStyle::with_template("    {spinner} {msg} {elapsed}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner());
    let workers: Vec<ProgressBar> = (0..n_workers)
        .map(|_| {
            let pb = bars.add(ProgressBar::new_spinner());
            pb.set_style(worker_style.clone());
            pb.set_message("idle");
            pb.enable_steady_tick(Duration::from_millis(120));
            pb
        })
        .collect();

    // A shared pull queue: each worker claims the next job by atomically
    // bumping `next`, so the N largest jobs are the first N claimed -- one
    // per worker, all running at once. This is what makes the largest-first
    // schedule observable on the per-worker spinners. Each worker owns
    // exactly one spinner, so the message updates need no further locking.
    let next = AtomicUsize::new(0);
    let job_count = jobs.len();
    let cells: Vec<ConfigRecord> = std::thread::scope(|scope| {
        let handles: Vec<_> = workers
            .iter()
            .map(|worker| {
                let (next, jobs, probed, overall) = (&next, &jobs, probed, &overall);
                scope.spawn(move || {
                    let mut done: Vec<(usize, ConfigRecord)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= job_count {
                            break;
                        }
                        let (p, l) = jobs[i];
                        worker.reset_elapsed();
                        worker.set_message(format!("{} {}", probed[p].name, levels[l].flag()));
                        let cell = deterministic_config(&probed[p], levels[l], ours_bin, opts);
                        overall.inc(1);
                        done.push((i, cell));
                    }
                    worker.set_message("idle");
                    done
                })
            })
            .collect();
        // Gather each worker's completed (job index, record) pairs and put
        // them back into job order so `cells` aligns with `jobs`.
        let mut all: Vec<(usize, ConfigRecord)> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("cachegrind worker panicked"))
            .collect();
        all.sort_by_key(|(i, _)| *i);
        all.into_iter().map(|(_, cell)| cell).collect()
    });
    for worker in &workers {
        worker.finish_and_clear();
    }
    overall.finish_and_clear();

    let mut records: Vec<ProgramRecord> = probed
        .iter()
        .map(|pr| ProgramRecord {
            name: pr.name.clone(),
            values: pr.values,
            verified: pr.verified,
            emitted_ir: None,
            configs: Vec::with_capacity(levels.len()),
        })
        .collect();
    for (&(p, _), cell) in jobs.iter().zip(cells) {
        records[p].configs.push(cell);
    }
    // Jobs ran largest-first, so restore level order within each program
    // ("o0" < "o2" < "o3" sorts correctly as strings).
    for record in &mut records {
        record.configs.sort_by(|a, b| a.level.cmp(&b.level));
    }
    records
}

/// The default run: probe the corpus, then the parallel deterministic
/// Cachegrind pass. Only the machine-independent counts (+ object size)
/// are recorded.
pub fn measure_deterministic_corpus(
    programs: &[&Program],
    levels: &[CodegenLevel],
    ours_bin: &Path,
    opts: &RecordOpts,
) -> color_eyre::Result<Vec<ProgramRecord>> {
    let probed = probe_all(programs);
    let ours_bin = ours_bin.to_string_lossy().into_owned();
    Ok(deterministic_records(&probed, levels, &ours_bin, opts))
}

/// The `--wall` competitive run: the deterministic Cachegrind pass first
/// (parallel), then augment each record in place with wall + peak RSS,
/// the in-process per-phase breakdown, and a clang reference line. Reuses
/// each program's already-staged bitcode, so nothing is staged or probed
/// twice, and the deterministic counts survive even a failed wall compile.
pub fn measure_competitive_corpus(
    programs: &[&Program],
    levels: &[CodegenLevel],
    ours_bin: &Path,
    opts: &RecordOpts,
    counters: &mut Counters,
) -> color_eyre::Result<Vec<ProgramRecord>> {
    let probed = probe_all(programs);
    let ours_bin = ours_bin.to_string_lossy().into_owned();
    let mut records = deterministic_records(&probed, levels, &ours_bin, opts);
    augment_with_wall(&probed, &mut records, levels, &ours_bin, opts, counters);
    Ok(records)
}

/// Add wall + peak RSS + in-process phases and a clang reference line to the
/// records the deterministic pass produced. Sequential and contention-
/// sensitive by design; owns its own progress bar. A program whose wall
/// augmentation errors keeps its deterministic counts and is noted, not
/// dropped.
fn augment_with_wall(
    probed: &[Probed],
    records: &mut [ProgramRecord],
    levels: &[CodegenLevel],
    ours_bin: &str,
    opts: &RecordOpts,
    counters: &mut Counters,
) {
    let bar = ProgressBar::new(probed.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("  wall [{bar:28}] {pos}/{len} {msg} ({elapsed})")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    bar.enable_steady_tick(Duration::from_millis(120));
    for (prog, record) in probed.iter().zip(records.iter_mut()) {
        bar.set_message(prog.name.clone());
        match augment_one(prog, record, levels, ours_bin, opts, counters, &bar) {
            Ok(()) => bar.println(format!(
                "  {} ... {} values{}",
                record.name,
                record.values,
                if record.verified { "" } else { " (UNVERIFIED)" }
            )),
            // The deterministic counts are already in `record`; a failed
            // wall compile just leaves this program without wall/clang data.
            Err(error) => bar.println(format!("  {} ... wall skipped ({error:#})", prog.name)),
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
}

/// Measure wall/RSS + the in-process phases for one program's ours configs
/// and its clang reference line, merging them into the record the
/// deterministic pass produced. Reuses the staged bitcode (no re-staging).
fn augment_one(
    prog: &Probed,
    record: &mut ProgramRecord,
    levels: &[CodegenLevel],
    ours_bin: &str,
    opts: &RecordOpts,
    counters: &mut Counters,
    bar: &ProgressBar,
) -> color_eyre::Result<()> {
    let io = tempfile::tempdir()?;
    let bc = prog.staged.path();
    for &level in levels {
        // Our in-process phases (mid+backend from the staged IR). The
        // steady tick repaints from a background thread and its string
        // formatting allocates into the process-global allocation
        // counters, so it is paused while phases are measured -- the
        // per-phase allocation counts stay deterministic.
        bar.disable_steady_tick();
        let run = measure_phases(
            bc,
            &prog.name,
            level.to_inkwell(),
            opts.warmup,
            opts.iters,
            counters,
        );
        bar.enable_steady_tick(Duration::from_millis(120));
        let run = run?;
        let phases = phase_records(&run);
        let passes = pass_records(&run);
        if record.emitted_ir.is_none() {
            record.emitted_ir = run.emitted_ir;
        }

        // ours wall, from the same staged bitcode the deterministic pass
        // compiled -- so its object size / counts stay valid.
        let ours_obj = io.path().join("ours_out.o");
        let ours_args = ours_compile_args(level, &ours_obj, bc);
        let ours = {
            let label = format!("ours {}", level.flag());
            let mut on_iter =
                |i, t| bar.set_message(format!("{} | {label} | iter {i}/{t}", prog.name));
            measure_end_to_end(
                ours_bin,
                &ours_args,
                opts,
                &ours_obj,
                io.path(),
                &mut on_iter,
            )?
        };
        merge_wall(record, level, ours, phases, passes);

        // clang reference line (a fresh config; clang carries no Cachegrind).
        let clang_out = io.path().join("clang_out.o");
        let clang_args = clang_compile_args(level, &clang_out, bc);
        let clang = {
            let label = format!("clang {}", level.flag());
            let mut on_iter =
                |i, t| bar.set_message(format!("{} | {label} | iter {i}/{t}", prog.name));
            measure_end_to_end(
                "clang",
                &clang_args,
                opts,
                &clang_out,
                io.path(),
                &mut on_iter,
            )?
        };
        // Pushed as each level completes (not batched), so a later level's
        // error keeps the clang lines already measured for earlier levels.
        record.configs.push(config_record(
            Compiler::Clang,
            level,
            clang,
            Vec::new(),
            Vec::new(),
        ));
    }
    Ok(())
}

/// Merge a wall measurement and its in-process phases into the ours config
/// the deterministic pass recorded for `level`, preserving that config's
/// Cachegrind counts and object size. A failed or timed-out wall compile
/// adds nothing -- the deterministic result stands, never erased.
fn merge_wall(
    record: &mut ProgramRecord,
    level: CodegenLevel,
    end: EndToEnd,
    phases: Vec<PhaseRecord>,
    passes: Vec<PassRecord>,
) {
    // Only merge onto a config the deterministic pass recorded as measured:
    // if that pass failed/timed out for this level, its verdict is
    // authoritative, and a wall compile that happens to succeed (e.g. it hit
    // no contention) must not decorate a `Failed` config with wall samples.
    let Some(cfg) = record.configs.iter_mut().find(|c| {
        matches!(c.compiler, Compiler::Ours)
            && matches!(c.status, ConfigStatus::Measured)
            && c.level == level.flag()
    }) else {
        return;
    };
    if let Measurement::Measured(samples) = end.measurement {
        cfg.end_to_end = subprocess_wall_samples(&samples);
        cfg.peak_rss_bytes = subprocess::median_peak_rss(&samples);
        cfg.phases = phases;
        cfg.passes = passes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::cachegrind::CachegrindStats;
    use crate::bench::measure::SampleFailure;

    fn failed(timed_out: bool, stderr: &str) -> EndToEnd {
        EndToEnd {
            measurement: Measurement::Failed(SampleFailure {
                timed_out,
                stderr: stderr.to_string(),
            }),
            // Present even on failure -> failed_config must drop it.
            object_size_bytes: Some(999),
        }
    }

    #[test]
    fn short_stderr_kept_whole() {
        assert_eq!(stderr_tail("error: boom\n"), "error: boom");
    }

    #[test]
    fn long_stderr_truncated_to_tail() {
        let input: String = (0..100).map(|i| format!("line {i}\n")).collect();
        let tail = stderr_tail(&input);
        assert!(tail.starts_with("... (truncated, last 40 lines)"));
        assert!(tail.ends_with("line 99"));
        assert!(!tail.contains("line 59")); // 100 lines, last 40 = 60..=99
        assert!(tail.contains("line 60"));
    }

    #[test]
    fn failed_config_records_status_and_error_not_data() {
        let record = config_record(
            Compiler::Ours,
            CodegenLevel::O0,
            failed(false, "boom"),
            vec![],
            vec![],
        );
        assert!(matches!(record.status, ConfigStatus::Failed));
        assert_eq!(record.error.as_deref(), Some("boom"));
        assert!(record.end_to_end.wall_ms.is_empty());
        assert_eq!(record.peak_rss_bytes, None);
        // A failed compile must not report an object size (the file on
        // disk, if any, belongs to a prior config) or any cachegrind count.
        assert_eq!(record.object_size_bytes, None);
        assert_eq!(record.cachegrind_ir, None);
        assert_eq!(record.cachegrind_ll_misses, None);
        assert_eq!(record.cachegrind_total_accesses, None);
        assert_eq!(record.cachegrind_estimated_cycles, None);
    }

    #[test]
    fn timeout_maps_to_timed_out_status() {
        let record = config_record(
            Compiler::Clang,
            CodegenLevel::O2,
            failed(true, "killed"),
            vec![],
            vec![],
        );
        assert!(matches!(record.status, ConfigStatus::TimedOut));
    }

    #[test]
    fn config_record_measured_clang_has_no_error_or_counts() {
        // config_record is the clang path: measured, with wall + object size
        // but never Cachegrind counts (only ours is simulated).
        let e = EndToEnd {
            measurement: Measurement::Measured(vec![SubprocessSample {
                wall: std::time::Duration::from_millis(5),
                peak_rss_bytes: Some(1024),
            }]),
            object_size_bytes: Some(200),
        };
        let record = config_record(Compiler::Clang, CodegenLevel::O3, e, vec![], vec![]);
        assert!(matches!(record.status, ConfigStatus::Measured));
        assert_eq!(record.error, None);
        assert_eq!(record.object_size_bytes, Some(200));
        assert_eq!(record.end_to_end.wall_ms.len(), 1);
        assert_eq!(record.cachegrind_ir, None);
    }

    #[test]
    fn measured_config_maps_all_cachegrind_fields() {
        // measured_config is the single Cachegrind -> record mapping (the
        // live path for ours configs in the deterministic pass).
        let record = measured_config(
            Compiler::Ours,
            CodegenLevel::O3,
            MeasuredConfig {
                end_to_end: MetricSamples::default(),
                peak_rss_bytes: None,
                object_size_bytes: Some(200),
                cachegrind: Some(CachegrindStats {
                    ir: 42,
                    ll_misses: 7,
                    total_accesses: 100,
                    estimated_cycles: 99,
                }),
            },
            vec![],
            vec![],
        );
        assert!(matches!(record.status, ConfigStatus::Measured));
        assert_eq!(record.error, None);
        assert_eq!(record.object_size_bytes, Some(200));
        assert_eq!(record.cachegrind_ir, Some(42));
        assert_eq!(record.cachegrind_ll_misses, Some(7));
        assert_eq!(record.cachegrind_total_accesses, Some(100));
        assert_eq!(record.cachegrind_estimated_cycles, Some(99));
    }

    #[test]
    fn merge_wall_preserves_cachegrind_and_ignores_failed_wall() {
        // A deterministic ours config already carries Cachegrind counts and
        // an object size, but no wall yet.
        let stats = CachegrindStats {
            ir: 5,
            ll_misses: 1,
            total_accesses: 50,
            estimated_cycles: 9,
        };
        let ours = measured_config(
            Compiler::Ours,
            CodegenLevel::O0,
            MeasuredConfig {
                end_to_end: MetricSamples::default(),
                peak_rss_bytes: None,
                object_size_bytes: Some(4096),
                cachegrind: Some(stats),
            },
            vec![],
            vec![],
        );
        let mut record = ProgramRecord {
            name: "prog".to_string(),
            values: 1,
            verified: true,
            emitted_ir: None,
            configs: vec![ours],
        };

        // A failed/timed-out wall compile must add nothing and erase nothing.
        let failed_wall = EndToEnd {
            measurement: Measurement::Failed(SampleFailure {
                timed_out: true,
                stderr: "boom".to_string(),
            }),
            object_size_bytes: None,
        };
        merge_wall(&mut record, CodegenLevel::O0, failed_wall, vec![], vec![]);
        let cfg = &record.configs[0];
        assert!(matches!(cfg.status, ConfigStatus::Measured));
        assert_eq!(
            cfg.cachegrind_ir,
            Some(5),
            "deterministic counts survive a failed wall"
        );
        assert_eq!(cfg.object_size_bytes, Some(4096));
        assert!(
            cfg.end_to_end.wall_ms.is_empty(),
            "no wall added from a failure"
        );

        // A measured wall compile fills wall + RSS while keeping the counts.
        let good_wall = EndToEnd {
            measurement: Measurement::Measured(vec![SubprocessSample {
                wall: std::time::Duration::from_millis(7),
                peak_rss_bytes: Some(2048),
            }]),
            object_size_bytes: None,
        };
        merge_wall(&mut record, CodegenLevel::O0, good_wall, vec![], vec![]);
        let cfg = &record.configs[0];
        assert_eq!(cfg.end_to_end.wall_ms.len(), 1);
        assert_eq!(cfg.peak_rss_bytes, Some(2048));
        assert_eq!(
            cfg.cachegrind_ir,
            Some(5),
            "counts still intact after a good wall"
        );
    }

    #[test]
    fn merge_wall_leaves_a_failed_deterministic_config_untouched() {
        // The deterministic pass failed this level; a wall compile that later
        // succeeds must not decorate the failed config with wall samples.
        let failed = failed_config(
            Compiler::Ours,
            CodegenLevel::O0,
            &SampleFailure {
                timed_out: false,
                stderr: "det boom".to_string(),
            },
            vec![],
            vec![],
        );
        let mut record = ProgramRecord {
            name: "prog".to_string(),
            values: 1,
            verified: true,
            emitted_ir: None,
            configs: vec![failed],
        };
        let good_wall = EndToEnd {
            measurement: Measurement::Measured(vec![SubprocessSample {
                wall: std::time::Duration::from_millis(3),
                peak_rss_bytes: Some(1024),
            }]),
            object_size_bytes: None,
        };
        merge_wall(&mut record, CodegenLevel::O0, good_wall, vec![], vec![]);
        let cfg = &record.configs[0];
        assert!(
            matches!(cfg.status, ConfigStatus::Failed),
            "status stays Failed"
        );
        assert!(cfg.error.is_some());
        assert!(
            cfg.end_to_end.wall_ms.is_empty(),
            "no wall attached to a failed config"
        );
    }
}
