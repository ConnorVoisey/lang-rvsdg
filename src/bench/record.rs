//! The durable per-run record -- the benchmark store's schema.
//!
//! One `RunRecord` per invocation is written as `runs/<epoch>-<sha>.json`
//! and never rewritten; the report reads the set. This module IS the
//! schema (the reviewers asked it be written down): every field is
//! documented here, `SCHEMA_VERSION` bumps on any breaking change, and
//! the report normalizes older versions on load.
//!
//! Metrics are stored as RAW SAMPLE VECTORS, not aggregates: the report
//! computes medians, error bars, and significance (Mann-Whitney) from
//! the samples, which a min/median/max summary could not support.
//! Counters are `None` when the kernel denied perf access or the config
//! was measured out-of-process (clang, and the whole-process RSS pass).

use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::WrapErr;
use serde::Serialize;

use crate::stats::EmittedIrStats;

/// Bumped on any breaking change to the record shape. The report reader
/// must handle every version it may encounter in a store.
///
/// v2: `ConfigRecord` gained `status` and `error`, so a compile that
/// failed or timed out is recorded (with its stderr) instead of appearing
/// as an empty config. A v1 reader treats an absent `status` as measured.
///
/// v3: `ConfigRecord` gained `cachegrind_ll_misses` and
/// `cachegrind_estimated_cycles` (Cachegrind now runs with cache
/// simulation on). Older runs simply lack the two fields.
///
/// v4: `ConfigRecord` gained `cachegrind_total_accesses` (Ir + Dr + Dw),
/// the denominator that turns the raw LL miss count into a miss rate.
pub const SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Serialize)]
pub struct RunRecord {
    pub schema_version: u32,
    pub meta: RunMeta,
    pub programs: Vec<ProgramRecord>,
}

/// Conditions the run was taken under -- so a number is reproducible and
/// comparable across time. `clang_version` matters because clang is the
/// reference line for every comparison; a clang upgrade shifts it.
#[derive(Debug, Serialize)]
pub struct RunMeta {
    pub timestamp_unix: u64,
    pub git_sha: String,
    pub git_dirty: bool,
    pub hostname: String,
    pub cpu_model: String,
    /// The CPU frequency governor, when readable; noisy cycles come from
    /// anything but `performance`.
    pub governor: Option<String>,
    pub clang_version: String,
    pub iters: u32,
    pub warmup: u32,
    /// True when `/usr/bin/time` was present, i.e. peak-RSS was measured.
    pub rss_available: bool,
}

#[derive(Debug, Serialize)]
pub struct ProgramRecord {
    pub name: String,
    /// Post-construction graph size, for context and normalization.
    pub values: usize,
    /// Correctness tier one: our pipeline's own `verify()` passed and the
    /// object built. (Output-match against clang is the runtime tier,
    /// added with runtime.)
    pub verified: bool,
    /// Emitted LLVM module shape (ours only, level-independent, so stored
    /// once per program rather than per config).
    pub emitted_ir: Option<EmittedIrStats>,
    pub configs: Vec<ConfigRecord>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum Compiler {
    Ours,
    Clang,
}

/// Whether the end-to-end compile for a config succeeded, failed, or was
/// killed at the timeout ceiling. A failed/timed-out config still has a
/// record (with `error`), so a broken compile is visible, not dropped.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum ConfigStatus {
    Measured,
    Failed,
    TimedOut,
}

/// One `{compiler, level}` cell of the matrix. `phases`/`passes` are
/// populated only for our in-process configs (clang is opaque); the
/// out-of-process `end_to_end`/`peak_rss`/`object_size` are populated
/// for both when the compile succeeded.
#[derive(Debug, Serialize)]
pub struct ConfigRecord {
    pub compiler: Compiler,
    /// Codegen level label: "o0" / "o1" / "o2" / "o3".
    pub level: String,
    /// End-to-end compile outcome. When not `Measured`, the sample-bearing
    /// fields below are empty/`None` and `error` holds the stderr tail.
    pub status: ConfigStatus,
    /// Tail of the failing compile's stderr; `None` when measured.
    pub error: Option<String>,
    /// Whole-compile wall from a subprocess (both compilers).
    pub end_to_end: MetricSamples,
    pub peak_rss_bytes: Option<u64>,
    pub object_size_bytes: Option<u64>,
    /// Simulated Cachegrind counts, so run-to-run noise is tiny (under
    /// ~0.05%, versus tens of percent for a hardware counter) -- a delta
    /// above that floor is a real code change, which makes these the
    /// usable regression signals. They count the invoked process only: for
    /// our driver that is our pipeline plus the in-process LLVM backend,
    /// EXCLUDING the clang/opt frontend it spawns. Only ours is measured
    /// under Cachegrind -- clang's are always `None` -- so these are a
    /// regression signal against ours' own baseline, not a cross-compiler
    /// comparison. Also `None` when Cachegrind is disabled
    /// (`--no-cachegrind`), valgrind is absent, or the run failed.
    ///
    /// `cachegrind_ir` is instructions executed; `cachegrind_ll_misses` is
    /// last-level cache misses (a deterministic locality proxy for the
    /// noisy hardware counter); `cachegrind_total_accesses` (Ir + Dr + Dw)
    /// is its denominator, so the report can show a size-independent miss
    /// rate; `cachegrind_estimated_cycles` is the cost model
    /// `L1_hits + 5*LL_hits + 35*RAM_hits` (a comparative index, not a
    /// literal cycle count).
    pub cachegrind_ir: Option<u64>,
    pub cachegrind_ll_misses: Option<u64>,
    pub cachegrind_total_accesses: Option<u64>,
    pub cachegrind_estimated_cycles: Option<u64>,
    /// Our per-phase in-process breakdown with hardware counters; empty
    /// for clang.
    pub phases: Vec<PhaseRecord>,
    /// Our optimise-pipeline per-pass wall; empty for clang.
    pub passes: Vec<PassRecord>,
}

#[derive(Debug, Serialize)]
pub struct PhaseRecord {
    pub phase: String,
    pub samples: MetricSamples,
}

#[derive(Debug, Serialize)]
pub struct PassRecord {
    pub name: String,
    pub wall_ms: Vec<f64>,
}

/// One iteration per element. Counter vectors are `None` when
/// unmeasured (no perf access / out-of-process); when `Some`, they have
/// the same length as `wall_ms`.
#[derive(Debug, Default, Serialize)]
pub struct MetricSamples {
    pub wall_ms: Vec<f64>,
    pub cycles: Option<Vec<u64>>,
    pub instructions: Option<Vec<u64>>,
    pub cache_misses: Option<Vec<u64>>,
    pub cache_references: Option<Vec<u64>>,
}

impl RunMeta {
    /// Snapshot the run conditions. Missing pieces degrade to a
    /// placeholder rather than fail -- a benchmark on a machine without
    /// git or `/proc/cpuinfo` should still record what it can.
    pub fn capture(iters: u32, warmup: u32, rss_available: bool) -> RunMeta {
        RunMeta {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            git_sha: git_output(&["rev-parse", "--short", "HEAD"])
                .unwrap_or_else(|| "nogit".to_string()),
            git_dirty: !git_output(&["status", "--porcelain"])
                .unwrap_or_default()
                .is_empty(),
            hostname: read_trimmed("/proc/sys/kernel/hostname").unwrap_or_default(),
            cpu_model: cpu_model().unwrap_or_default(),
            governor: crate::bench::measure::cpu_governor(),
            clang_version: clang_version().unwrap_or_default(),
            iters,
            warmup,
            rss_available,
        }
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn read_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn cpu_model() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    info.lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|rest| rest.split(':').nth(1))
        .map(|name| name.trim().to_string())
}

fn clang_version() -> Option<String> {
    let out = Command::new("clang").arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next().map(|line| line.trim().to_string())
}

/// Write the record as `runs/<epoch>-<sha>.json` under `dir` and return
/// its path. Lexicographic filename order is time order.
pub fn write_run(dir: &Path, record: &RunRecord) -> color_eyre::Result<PathBuf> {
    let runs = dir.join("runs");
    std::fs::create_dir_all(&runs)?;
    // The epoch-second timestamp is only 1s resolution, so two runs on the
    // same commit within a second would share a name and the second would
    // clobber the first. Append `-2`, `-3`, ... to the first free name so a
    // durable record is never overwritten (intra-second order is arbitrary
    // anyway -- the timestamp cannot distinguish them).
    let base = format!("{:010}-{}", record.meta.timestamp_unix, record.meta.git_sha);
    let mut path = runs.join(format!("{base}.json"));
    let mut n = 2;
    while path.exists() {
        path = runs.join(format!("{base}-{n}.json"));
        n += 1;
    }
    let file = std::fs::File::create(&path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, record)?;
    std::io::Write::flush(&mut writer)?;
    Ok(path)
}

/// Regenerate `data.js` -- the aggregate the static report loads over
/// `file://` (a double-clicked page cannot `fetch` sibling JSON). The
/// per-run JSONs stay the source of truth; this is a cheap derived
/// concatenation of them, in filename (time) order.
pub fn regenerate_data_js(dir: &Path) -> color_eyre::Result<()> {
    let runs = dir.join("runs");
    // Propagate read failures rather than swallow them: silently listing a
    // subset (or none) of the durable store would rewrite data.js from a
    // partial view that the report then loads as the whole truth.
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&runs).wrap_err_with(|| format!("reading {}", runs.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            files.push(path);
        }
    }
    files.sort();

    let mut out = String::from("window.BENCH_RUNS = [\n");
    for (index, path) in files.iter().enumerate() {
        let json = std::fs::read_to_string(path)?;
        out.push_str(json.trim_end());
        if index + 1 < files.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("];\n");
    std::fs::write(dir.join("data.js"), out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run store that cannot be listed must error, not silently rewrite
    /// data.js from an empty view. `runs` is created as a file so
    /// `read_dir` fails deterministically (chmod would be a no-op as root).
    #[test]
    fn regenerate_errors_when_runs_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("runs"), b"not a directory").unwrap();
        assert!(regenerate_data_js(dir.path()).is_err());
    }

    /// Two runs on the same commit within one second must not clobber each
    /// other -- the second gets a disambiguated name.
    #[test]
    fn write_run_never_overwrites_same_second() {
        let dir = tempfile::tempdir().unwrap();
        let record = || RunRecord {
            schema_version: SCHEMA_VERSION,
            meta: RunMeta {
                timestamp_unix: 42,
                git_sha: "abc".to_string(),
                git_dirty: false,
                hostname: "h".to_string(),
                cpu_model: "c".to_string(),
                governor: None,
                clang_version: "x".to_string(),
                iters: 1,
                warmup: 0,
                rss_available: false,
            },
            programs: Vec::new(),
        };
        let p1 = write_run(dir.path(), &record()).unwrap();
        let p2 = write_run(dir.path(), &record()).unwrap();
        assert_ne!(p1, p2);
        assert!(p1.exists() && p2.exists());
        let count = std::fs::read_dir(dir.path().join("runs")).unwrap().count();
        assert_eq!(count, 2);
    }

    /// Happy path: every run JSON is inlined, in filename (time) order.
    #[test]
    fn regenerate_inlines_runs_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let runs = dir.path().join("runs");
        std::fs::create_dir_all(&runs).unwrap();
        std::fs::write(runs.join("0000000002-b.json"), r#"{"n":2}"#).unwrap();
        std::fs::write(runs.join("0000000001-a.json"), r#"{"n":1}"#).unwrap();
        std::fs::write(runs.join("ignore.txt"), "not json").unwrap();

        regenerate_data_js(dir.path()).unwrap();
        let js = std::fs::read_to_string(dir.path().join("data.js")).unwrap();
        assert_eq!(js, "window.BENCH_RUNS = [\n{\"n\":1},\n{\"n\":2}\n];\n");
    }
}
