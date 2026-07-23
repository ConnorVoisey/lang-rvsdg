//! Deterministic instruction, cache, and cost counting under Cachegrind.
//!
//! Wall time is noisy and machine-dependent; hardware perf counters are
//! too (last-level cache misses especially -- run-to-run swings of tens of
//! percent on a busy machine, with no code change). Cachegrind instead
//! runs the compile on a CPU simulator, so its counts barely move
//! run-to-run (empirically under ~0.05%, versus tens of percent for the
//! hardware counter). They are not bit-identical -- the small residual is
//! nondeterminism in the measured program itself (e.g. randomly-seeded
//! `HashMap` iteration order affecting the work done), not the machine --
//! but a delta above that floor is a real code change, which is what makes
//! these the usable regression signals.
//!
//! Three numbers are derived:
//! - `ir`: instructions executed.
//! - `ll_misses`: last-level cache misses -- a deterministic locality
//!   proxy that replaces the noisy hardware cache-miss counter.
//! - `estimated_cycles`: the iai/cachegrind cost model
//!   `L1_hits + 5*LL_hits + 35*RAM_hits`, a single deterministic proxy for
//!   how expensive the compile is (instruction work plus memory-stall
//!   penalty). It is a simplified model -- no instruction-level
//!   parallelism, out-of-order execution, or branch prediction -- so it
//!   OVER-estimates absolute cycles and is a comparative index, not a
//!   literal cycle/time prediction.
//!
//! The cache model is PINNED to a fixed reference cache (not the host's),
//! so counts are comparable across machines: they model that cache, not
//! this CPU. The cost is speed -- cache simulation is the slow part of
//! Cachegrind, ~10-30x native -- so this runs ONCE per config (the counts
//! are exact, no iterations).
//!
//! `--trace-children=no` (valgrind's default, set explicitly) counts ONLY
//! the invoked process. For our driver that excludes the clang/opt
//! frontend it spawns as children -- and includes the LLVM backend, which
//! runs in-process -- so the counts isolate OUR compiler's work. Only ours
//! is measured under Cachegrind (clang is not), so these counts are a
//! regression signal against ours' own baseline, never a cross-compiler
//! one.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::bench::measure::run_with_timeout;

// A fixed reference cache so counts are comparable across machines: 32 KiB
// 8-way L1 (instruction and data), 8 MiB 16-way last level, 64 B lines.
// `size,associativity,line_size`.
const L1_CACHE: &str = "32768,8,64";
const LL_CACHE: &str = "8388608,16,64";

/// Deterministic counts from one Cachegrind run.
#[derive(Debug, Clone, Copy)]
pub struct CachegrindStats {
    /// Instructions executed.
    pub ir: u64,
    /// Last-level cache misses (a deterministic locality proxy).
    pub ll_misses: u64,
    /// Total memory references (Ir + Dr + Dw): the denominator for the LL
    /// miss rate, so the size-independent locality figure can be recovered.
    pub total_accesses: u64,
    /// Cost model `L1_hits + 5*LL_hits + 35*RAM_hits` -- a comparative
    /// index, not a literal cycle count.
    pub estimated_cycles: u64,
}

/// Whether `valgrind` is on PATH. When it is not, the benchmark still runs
/// and the counts read `None`.
pub fn available() -> bool {
    Command::new("valgrind")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run `program args...` under Cachegrind once and return its deterministic
/// counts. `None` if valgrind is missing, the command fails, or the output
/// cannot be parsed -- a missing count is not a data point, never a
/// fabricated one.
pub fn measure(
    program: &str,
    args: &[String],
    timeout: Duration,
    io_dir: &Path,
) -> Option<CachegrindStats> {
    // A distinct out-file per call so ours and clang in the same io_dir do
    // not clobber each other's summary.
    let out_file = tempfile::Builder::new()
        .prefix("cachegrind.")
        .suffix(".out")
        .tempfile_in(io_dir)
        .ok()?;

    let mut cmd = Command::new("valgrind");
    cmd.arg("--tool=cachegrind")
        .arg("--cache-sim=yes")
        .arg("--branch-sim=no")
        .arg("--trace-children=no")
        .arg(format!("--I1={L1_CACHE}"))
        .arg(format!("--D1={L1_CACHE}"))
        .arg(format!("--LL={LL_CACHE}"))
        .arg(format!(
            "--cachegrind-out-file={}",
            out_file.path().display()
        ))
        .arg(program)
        .args(args);

    let outcome = run_with_timeout(&mut cmd, timeout, io_dir).ok()?;
    if !outcome.success() {
        return None;
    }
    let contents = std::fs::read_to_string(out_file.path()).ok()?;
    parse_summary(&contents)
}

/// Parse Cachegrind's `summary:` line (with cache simulation on, no
/// thousands separators) into the derived counts. The nine fields are, in
/// order: `Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw`.
fn parse_summary(contents: &str) -> Option<CachegrindStats> {
    let line = contents.lines().find_map(|l| l.strip_prefix("summary:"))?;
    let f: Vec<u64> = line
        .split_whitespace()
        .map(|s| s.parse().ok())
        .collect::<Option<Vec<u64>>>()?;
    if f.len() < 9 {
        return None;
    }

    let ir = f[0];
    let l1_misses = f[1] + f[4] + f[7]; // I1mr + D1mr + D1mw
    let ll_misses = f[2] + f[5] + f[8]; // ILmr + DLmr + DLmw
    let total_accesses = f[0] + f[3] + f[6]; // Ir + Dr + Dw

    // Each access is a hit at some level; the misses cascade L1 -> LL ->
    // RAM. Weights are the iai/cachegrind model (1 / 5 / 35 cycles).
    let l1_hits = total_accesses.saturating_sub(l1_misses);
    let ll_hits = l1_misses.saturating_sub(ll_misses);
    let ram_hits = ll_misses;
    let estimated_cycles = l1_hits + 5 * ll_hits + 35 * ram_hits;

    Some(CachegrindStats {
        ir,
        ll_misses,
        total_accesses,
        estimated_cycles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_summary_with_cache_sim() {
        // summary: Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw
        let stats = parse_summary("summary: 12345 100 20 4000 30 5 2000 15 3\n").unwrap();
        assert_eq!(stats.ir, 12345);
        assert_eq!(stats.ll_misses, 28); // 20 + 5 + 3
        assert_eq!(stats.total_accesses, 18345); // 12345 + 4000 + 2000
        // l1_misses = 100+30+15 = 145; total = 12345+4000+2000 = 18345
        // l1_hits = 18200, ll_hits = 145-28 = 117, ram_hits = 28
        // 18200 + 5*117 + 35*28 = 18200 + 585 + 980
        assert_eq!(stats.estimated_cycles, 19765);
    }

    #[test]
    fn too_few_fields_is_none() {
        // A cache-sim-off summary has a single field; we require the full
        // nine, so it is treated as unusable rather than misparsed.
        assert_eq!(parse_summary("summary: 171381\n").map(|s| s.ir), None);
    }

    #[test]
    fn missing_summary_is_none() {
        assert!(parse_summary("desc: I1 cache: none\nevents: Ir\n").is_none());
    }
}
