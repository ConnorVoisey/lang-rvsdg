//! Differential-testing / fuzzing driver for the RVSDG compiler.
//!
//! For each C input it compiles two binaries -- one with the RVSDG compiler
//! (`lang-rvsdg`, this crate's main binary) and one with `clang` as the
//! reference -- runs both, and compares their stdout and exit code. With
//! `--count` it instead fuzzes: it generates programs with `csmith` and
//! runs the same comparison in a loop, saving any failing input.
//!
//! Everything is run as a subprocess with a timeout, so a compiler crash,
//! an `abort()`, or an infinite loop is isolated and reported rather than
//! taking the driver down -- and a slow/hanging compile is itself a finding.
//!
//! Outcomes:
//!   - PASS         -- same stdout and exit code
//!   - MISMATCH     -- different stdout/exit (the real correctness bug)
//!   - ICE          -- the RVSDG compiler exited nonzero (unsupported feature / crash)
//!   - COMPILE-SLOW -- the RVSDG compile exceeded the timeout (possible hang)
//!   - RUN-TIMEOUT  -- the RVSDG-produced binary hung
//!   - (clang-fail) -- the reference didn't compile; the input is skipped

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// csmith flags used when none are supplied via `--csmith-arg`. A
/// conservative starting point that still exercises a lot; tune freely.
const DEFAULT_CSMITH_ARGS: &[&str] = &[
    "--no-packed-struct",
    "--no-volatiles",
    "--no-volatile-pointers",
];

#[derive(Parser, Debug)]
#[command(
    name = "difftest",
    about = "RVSDG vs clang differential tester / csmith fuzzer"
)]
struct Args {
    /// C files to test. If none are given, fuzz with csmith (see --count).
    inputs: Vec<PathBuf>,

    /// Header search paths passed to both compilers (repeatable).
    #[arg(
        short = 'I',
        long = "include",
        value_name = "DIR",
        default_value = "/usr/include/csmith-2.3.0"
    )]
    include: Vec<String>,

    /// Number of csmith programs to fuzz when no inputs are given.
    #[arg(long, default_value_t = 100)]
    count: u64,

    /// Base csmith seed; iteration `i` uses `seed + i`. Defaults to a
    /// time-derived value (printed at startup so runs are reproducible).
    #[arg(long)]
    seed: Option<u64>,

    /// Extra csmith arguments (repeatable). Overrides the built-in default set.
    #[arg(long = "csmith-arg", value_name = "ARG")]
    csmith_arg: Vec<String>,

    /// Compile timeout in seconds; a slower RVSDG compile is a finding.
    #[arg(long, default_value_t = 30)]
    compile_timeout: u64,

    /// Per-program run timeout in seconds.
    #[arg(long, default_value_t = 10)]
    run_timeout: u64,

    /// Directory where failing inputs are saved.
    #[arg(long, default_value = "difftest-findings")]
    findings: PathBuf,

    /// Path to the RVSDG compiler binary (defaults to the sibling `lang-rvsdg`).
    #[arg(long)]
    cc: Option<PathBuf>,

    /// Number of parallel worker threads for fuzzing. 0 (the default) uses
    /// every available core.
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

/// The result of compiling one binary with the RVSDG compiler.
enum Compile {
    Ok {
        elapsed: Duration,
    },
    /// Compiler exited nonzero (unsupported feature, panic, ...).
    Ice {
        elapsed: Duration,
        stderr: String,
    },
    /// Compile exceeded the timeout (possible non-termination).
    Timeout {
        elapsed: Duration,
    },
}

/// The result of running a produced binary.
enum Run {
    Ok { stdout: Vec<u8>, code: Option<i32> },
    Timeout,
}

enum Outcome {
    Pass {
        rvsdg_compile: Duration,
        clang_compile: Duration,
        rvsdg_run: Duration,
        clang_run: Duration,
    },
    Mismatch {
        rvsdg: (Vec<u8>, Option<i32>),
        clang: (Vec<u8>, Option<i32>),
    },
    Ice {
        elapsed: Duration,
        stderr: String,
    },
    CompileSlow {
        elapsed: Duration,
    },
    RunTimeout,
    ClangCompileFail,
}

impl Outcome {
    fn is_failure(&self) -> bool {
        !matches!(self, Outcome::Pass { .. } | Outcome::ClangCompileFail)
    }
    fn label(&self) -> &'static str {
        match self {
            Outcome::Pass { .. } => "PASS",
            Outcome::Mismatch { .. } => "MISMATCH",
            Outcome::Ice { .. } => "ICE",
            Outcome::CompileSlow { .. } => "COMPILE-SLOW",
            Outcome::RunTimeout => "RUN-TIMEOUT",
            Outcome::ClangCompileFail => "clang-fail",
        }
    }
}

/// Spawn `cmd` and wait up to `timeout`, killing it if it runs over.
/// Returns the exit status (None if it timed out) and the wall-clock time.
fn spawn_timed(
    mut cmd: Command,
    timeout: Duration,
) -> std::io::Result<(Option<std::process::ExitStatus>, Duration)> {
    let start = Instant::now();
    let mut child = cmd.spawn()?;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok((Some(status), start.elapsed()));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Ok((None, start.elapsed()));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn compile_rvsdg(
    cc: &Path,
    c_file: &Path,
    out: &Path,
    args: &Args,
    tmp: &Path,
) -> std::io::Result<Compile> {
    let err_path = tmp.join("rvsdg.stderr");
    let err_file = fs::File::create(&err_path)?;
    let mut cmd = Command::new(cc);
    cmd.arg("--output").arg(out).arg("-q");
    for dir in &args.include {
        cmd.arg("-I").arg(dir);
    }
    cmd.arg(c_file);
    // Plain-text diagnostics (no ANSI) so saved finding logs stay readable.
    cmd.env("NO_COLOR", "1");
    cmd.stdout(Stdio::null()).stderr(err_file);
    let (status, elapsed) = spawn_timed(cmd, Duration::from_secs(args.compile_timeout))?;
    Ok(match status {
        None => Compile::Timeout { elapsed },
        Some(s) if s.success() && out.exists() => Compile::Ok { elapsed },
        Some(_) => Compile::Ice {
            elapsed,
            stderr: fs::read_to_string(&err_path).unwrap_or_default(),
        },
    })
}

fn compile_clang(c_file: &Path, out: &Path, args: &Args) -> std::io::Result<bool> {
    let mut cmd = Command::new("clang");
    // `-O0`: the RVSDG path does no optimisation yet, so an unoptimised clang
    // is the apples-to-apples baseline -- both the compile-time and runtime
    // comparisons then reflect codegen, not clang's optimiser doing extra work.
    cmd.args(["-O0", "-w"]);
    for dir in &args.include {
        cmd.arg("-I").arg(dir);
    }
    cmd.arg(c_file).arg("-o").arg(out);
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let (status, _) = spawn_timed(cmd, Duration::from_secs(args.compile_timeout))?;
    Ok(status.map(|s| s.success()).unwrap_or(false) && out.exists())
}

fn run_binary(bin: &Path, timeout: Duration, tmp: &Path) -> std::io::Result<(Run, Duration)> {
    // Redirect the program's stdout to a file (not a pipe) so a chatty
    // program can't deadlock on a full pipe buffer.
    let out_path = tmp.join("run.stdout");
    let out_file = fs::File::create(&out_path)?;
    let mut cmd = Command::new(bin);
    cmd.stdout(out_file)
        .stderr(Stdio::null())
        .stdin(Stdio::null());
    let (status, elapsed) = spawn_timed(cmd, timeout)?;
    Ok(match status {
        None => (Run::Timeout, elapsed),
        Some(s) => (
            Run::Ok {
                stdout: fs::read(&out_path).unwrap_or_default(),
                code: s.code(),
            },
            elapsed,
        ),
    })
}

fn differential_test(
    cc: &Path,
    c_file: &Path,
    args: &Args,
    tmp: &Path,
) -> std::io::Result<Outcome> {
    let rvsdg_bin = tmp.join("rvsdg_out");
    let clang_bin = tmp.join("clang_out");
    let _ = fs::remove_file(&rvsdg_bin);

    let rvsdg = compile_rvsdg(cc, c_file, &rvsdg_bin, args, tmp)?;
    let rvsdg_compile = match &rvsdg {
        Compile::Timeout { elapsed } => return Ok(Outcome::CompileSlow { elapsed: *elapsed }),
        Compile::Ice { elapsed, stderr } => {
            return Ok(Outcome::Ice {
                elapsed: *elapsed,
                stderr: stderr.clone(),
            });
        }
        Compile::Ok { elapsed, .. } => *elapsed,
    };

    // Reference compile. If clang can't build it, the input is not our bug.
    let clang_start = Instant::now();
    if !compile_clang(c_file, &clang_bin, args)? {
        return Ok(Outcome::ClangCompileFail);
    }
    let clang_compile = clang_start.elapsed();

    let run_timeout = Duration::from_secs(args.run_timeout);
    let (r_run, r_dur) = run_binary(&rvsdg_bin, run_timeout, tmp)?;
    let r = match r_run {
        Run::Timeout => return Ok(Outcome::RunTimeout),
        Run::Ok { stdout, code } => (stdout, code),
    };
    let (c_run, c_dur) = run_binary(&clang_bin, run_timeout, tmp)?;
    let c = match c_run {
        Run::Timeout => return Ok(Outcome::RunTimeout),
        Run::Ok { stdout, code } => (stdout, code),
    };

    if r == c {
        Ok(Outcome::Pass {
            rvsdg_compile,
            clang_compile,
            rvsdg_run: r_dur,
            clang_run: c_dur,
        })
    } else {
        Ok(Outcome::Mismatch { rvsdg: r, clang: c })
    }
}

/// Pull the most informative one-liner out of a failed compile's stderr:
/// the panic message (color-eyre prints `Message:  ...`) or the propagated
/// `Error:` text, skipping boilerplate.
/// Drop ANSI escape sequences (`ESC [ ... m`) so saved/printed reasons are
/// plain text regardless of the compiler's coloring.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for d in chars.by_ref() {
                if d == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn ice_reason(stderr: &str) -> String {
    let stderr = strip_ansi(stderr);
    let lines: Vec<&str> = stderr.lines().collect();
    for l in &lines {
        if let Some(rest) = l.trim().strip_prefix("Message:") {
            return rest.trim().to_string();
        }
    }
    if let Some(pos) = lines
        .iter()
        .position(|l| l.trim_start().starts_with("Error:"))
    {
        for l in &lines[pos..] {
            let t = l
                .trim()
                .trim_start_matches("Error:")
                .trim()
                .trim_start_matches(|c: char| c.is_ascii_digit() || c == ':')
                .trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    lines
        .iter()
        .map(|l| l.trim())
        .find(|l| {
            !l.is_empty()
                && !l.starts_with("Run with")
                && !l.starts_with("Backtrace")
                && !l.starts_with("stack backtrace")
                && !l.starts_with("note:")
        })
        .unwrap_or("(no message)")
        .to_string()
}

fn default_cc() -> PathBuf {
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    p.push("lang-rvsdg");
    p
}

fn run_csmith(seed: u64, out: &Path, args: &Args) -> bool {
    let mut cmd = Command::new("csmith");
    cmd.arg("--seed").arg(seed.to_string());
    let extra: Vec<String> = if args.csmith_arg.is_empty() {
        DEFAULT_CSMITH_ARGS.iter().map(|s| s.to_string()).collect()
    } else {
        args.csmith_arg.clone()
    };
    cmd.args(&extra);
    cmd.arg("-o").arg(out);
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

fn save_finding(c_file: &Path, tag: &str, outcome: &Outcome, dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let dest = dir.join(format!("{tag}.c"));
    fs::copy(c_file, &dest)?;
    if let Outcome::Ice { stderr, .. } = outcome {
        fs::write(dir.join(format!("{tag}.stderr")), stderr)?;
    }
    Ok(())
}

/// `a / b` as an f64, guarding the zero-duration denominator (sub-millisecond
/// runtimes round to zero often enough on trivial programs to matter).
fn ratio(a: Duration, b: Duration) -> f64 {
    let b = b.as_secs_f64();
    if b == 0.0 {
        f64::NAN
    } else {
        a.as_secs_f64() / b
    }
}

#[derive(Default)]
struct Stats {
    total: u64,
    pass: u64,
    mismatch: u64,
    ice: u64,
    compile_slow: u64,
    run_timeout: u64,
    clang_fail: u64,
    /// Summed over passing programs, so the aggregate compares rvsdg vs clang
    /// on the same corpus (the per-program ratio is reported by `report_single`).
    rvsdg_compile_sum: Duration,
    clang_compile_sum: Duration,
    rvsdg_run_sum: Duration,
    clang_run_sum: Duration,
    /// (compile time, tag) for the slowest RVSDG compiles seen.
    slowest: Vec<(Duration, String)>,
}

impl Stats {
    fn record(&mut self, tag: &str, outcome: &Outcome) {
        self.total += 1;
        match outcome {
            Outcome::Pass {
                rvsdg_compile,
                clang_compile,
                rvsdg_run,
                clang_run,
            } => {
                self.pass += 1;
                self.rvsdg_compile_sum += *rvsdg_compile;
                self.clang_compile_sum += *clang_compile;
                self.rvsdg_run_sum += *rvsdg_run;
                self.clang_run_sum += *clang_run;
                self.note_compile(*rvsdg_compile, tag);
            }
            Outcome::Mismatch { .. } => self.mismatch += 1,
            Outcome::Ice { elapsed, .. } => {
                self.ice += 1;
                self.note_compile(*elapsed, tag);
            }
            Outcome::CompileSlow { elapsed } => {
                self.compile_slow += 1;
                self.note_compile(*elapsed, tag);
            }
            Outcome::RunTimeout => self.run_timeout += 1,
            Outcome::ClangCompileFail => self.clang_fail += 1,
        }
    }
    fn note_compile(&mut self, d: Duration, tag: &str) {
        self.slowest.push((d, tag.to_string()));
        self.slowest.sort_by(|a, b| b.0.cmp(&a.0));
        self.slowest.truncate(5);
    }
    fn print(&self) {
        println!("\n=== summary ({} programs) ===", self.total);
        println!("  pass:          {}", self.pass);
        println!("  MISMATCH:      {}", self.mismatch);
        println!("  ICE:           {}", self.ice);
        println!("  compile-slow:  {}", self.compile_slow);
        println!("  run-timeout:   {}", self.run_timeout);
        println!("  clang-fail:    {}", self.clang_fail);
        if self.pass > 0 {
            let n = self.pass as f64;
            let mean = |d: Duration| d.as_secs_f64() / n;
            println!("  over {} passing programs (mean per program):", self.pass);
            println!(
                "    compile:  rvsdg {:.3}s  clang {:.3}s  ({:.1}x slower)",
                mean(self.rvsdg_compile_sum),
                mean(self.clang_compile_sum),
                ratio(self.rvsdg_compile_sum, self.clang_compile_sum),
            );
            println!(
                "    run:      rvsdg {:.3}s  clang {:.3}s  ({:.1}x slower)",
                mean(self.rvsdg_run_sum),
                mean(self.clang_run_sum),
                ratio(self.rvsdg_run_sum, self.clang_run_sum),
            );
        }
        if !self.slowest.is_empty() {
            println!("  slowest RVSDG compiles:");
            for (d, tag) in &self.slowest {
                println!("    {:>8.3}s  {tag}", d.as_secs_f64());
            }
        }
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let cc = args.cc.clone().unwrap_or_else(default_cc);
    if !cc.exists() {
        eprintln!(
            "RVSDG compiler not found at {} (build it, or pass --cc)",
            cc.display()
        );
        std::process::exit(2);
    }

    if !args.inputs.is_empty() {
        // Explicit inputs: run sequentially (the order and combined exit
        // code are the point) in a single shared scratch dir.
        let tmp = tempfile::tempdir()?;
        let mut failures = 0u32;
        for input in &args.inputs {
            let outcome = differential_test(&cc, input, &args, tmp.path())?;
            report_single(input, &outcome);
            if outcome.is_failure() {
                failures += 1;
            }
        }
        std::process::exit(if failures == 0 { 0 } else { 1 });
    }

    fuzz(&cc, &args)
}

/// Fuzz `args.count` csmith programs in parallel across all available cores
/// (or `--jobs` of them). Each seed is fully independent -- its own csmith
/// program, its own compiles, its own run -- so the only shared state is the
/// findings directory (distinct filenames per seed) and stdout. Each worker
/// gets a private temp dir so the fixed-name scratch files
/// (`rvsdg_out`, `clang_out`, ...) can't collide between threads.
fn fuzz(cc: &Path, args: &Args) -> std::io::Result<()> {
    if args.jobs > 0 {
        // Best-effort: ignore the error if the global pool is already set.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global();
    }

    let base = args.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    println!(
        "fuzzing {} programs from seed {base} on {} threads",
        args.count,
        rayon::current_num_threads(),
    );

    // Each worker returns its seed tag and outcome; csmith-generation
    // failures and IO errors are reported inline and dropped (None).
    let results: Vec<(String, Outcome)> = (0..args.count)
        .into_par_iter()
        .filter_map(|i| {
            let seed = base.wrapping_add(i);
            let tag = seed.to_string();
            let tmp = match tempfile::tempdir() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("seed {seed}: could not create temp dir: {e}");
                    return None;
                }
            };
            let cfile = tmp.path().join("fuzz.c");
            if !run_csmith(seed, &cfile, args) {
                eprintln!("seed {seed}: csmith failed to generate");
                return None;
            }
            let outcome = match differential_test(cc, &cfile, args, tmp.path()) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("seed {seed}: differential test error: {e}");
                    return None;
                }
            };
            if outcome.is_failure() {
                // One whole line per finding (ICE reason condensed to a
                // single line) so concurrent worker output stays readable.
                match &outcome {
                    Outcome::Ice { stderr, .. } => {
                        println!("seed {seed}: ICE -- {}", ice_reason(stderr))
                    }
                    other => println!("seed {seed}: {}", other.label()),
                }
                if let Err(e) = save_finding(&cfile, &tag, &outcome, &args.findings) {
                    eprintln!("seed {seed}: could not save finding: {e}");
                }
            }
            Some((tag, outcome))
        })
        .collect();

    // Aggregate sequentially: Stats ordering/ranking is deterministic and
    // independent of the order workers finished in.
    let mut stats = Stats::default();
    for (tag, outcome) in &results {
        stats.record(tag, outcome);
    }
    stats.print();
    if stats.mismatch + stats.ice + stats.compile_slow + stats.run_timeout > 0 {
        println!("findings saved in {}", args.findings.display());
    }
    Ok(())
}

fn report_single(input: &Path, outcome: &Outcome) {
    match outcome {
        Outcome::Pass {
            rvsdg_compile,
            clang_compile,
            rvsdg_run,
            clang_run,
        } => println!(
            "PASS  {}  compile: rvsdg {:.3}s / clang {:.3}s ({:.1}x)  run: rvsdg {:.3}s / clang {:.3}s ({:.1}x)",
            input.display(),
            rvsdg_compile.as_secs_f64(),
            clang_compile.as_secs_f64(),
            ratio(*rvsdg_compile, *clang_compile),
            rvsdg_run.as_secs_f64(),
            clang_run.as_secs_f64(),
            ratio(*rvsdg_run, *clang_run),
        ),
        Outcome::Mismatch { rvsdg, clang } => {
            println!("MISMATCH  {}", input.display());
            println!(
                "  rvsdg: exit={:?} stdout={:?}",
                rvsdg.1,
                String::from_utf8_lossy(&rvsdg.0)
            );
            println!(
                "  clang: exit={:?} stdout={:?}",
                clang.1,
                String::from_utf8_lossy(&clang.0)
            );
        }
        Outcome::Ice { elapsed, stderr } => {
            println!("ICE  {}  ({:.3}s)", input.display(), elapsed.as_secs_f64());
            println!("  {}", ice_reason(stderr));
        }
        Outcome::CompileSlow { elapsed } => {
            println!(
                "COMPILE-SLOW  {}  ({:.1}s, timed out)",
                input.display(),
                elapsed.as_secs_f64()
            )
        }
        Outcome::RunTimeout => println!("RUN-TIMEOUT  {}", input.display()),
        Outcome::ClangCompileFail => println!("clang-fail  {} (skipped)", input.display()),
    }
}
