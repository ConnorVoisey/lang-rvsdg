//! Out-of-process measurement of a compile: wall time, peak resident set
//! size, and output size. Used for the metrics an in-process measurement
//! cannot give per config -- clang (opaque) needs it, and even for our
//! own compiler peak RSS must come from a subprocess because a process's
//! peak is a monotonic high-water mark with no reset, so one process
//! cannot report a per-config peak.
//!
//! Peak RSS comes from wrapping the command in `/usr/bin/time -v` and
//! reading "Maximum resident set size". On Linux `wait4`'s `ru_maxrss`
//! folds the peak across the whole reaped process TREE, not just the
//! direct child, so this covers the subprocesses each compiler spawns --
//! our frontend shell-out to clang/opt, and clang's own `cc1` worker --
//! as well as the in-process LLVM C++ heap the Rust counting allocator
//! cannot see. That makes it symmetric and whole-tree for both sides
//! (verified: clang reports the same peak whether `cc1` is forked or
//! `-fintegrated-cc1`), and it needs no libc. When `/usr/bin/time` is
//! absent the command still runs and wall is timed; RSS reads `None`.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::bench::measure::{SampleFailure, median, run_with_timeout};

const GNU_TIME: &str = "/usr/bin/time";

#[derive(Debug, Clone, Copy)]
pub struct SubprocessSample {
    pub wall: Duration,
    /// `None` when `/usr/bin/time` was unavailable or its output could
    /// not be parsed.
    pub peak_rss_bytes: Option<u64>,
}

/// The result of measuring a compile across its runs: samples when every
/// run succeeded, or the first failure (with its stderr and whether it
/// was a timeout) so the caller can record it rather than discard it.
#[derive(Debug)]
pub enum Measurement {
    Measured(Vec<SubprocessSample>),
    Failed(SampleFailure),
}

/// Run `program args...` once under `/usr/bin/time -v` (when available),
/// timing wall from here and parsing peak RSS from time's report. The
/// GNU time wrapper's own overhead (~1ms) is inside the wall figure --
/// negligible against compiles measured in tens of milliseconds. `Err`
/// carries the run's stderr and whether it was killed at the timeout.
fn run_once(
    program: &str,
    args: &[String],
    timeout: Duration,
    io_dir: &Path,
    have_gnu_time: bool,
) -> std::io::Result<Result<SubprocessSample, SampleFailure>> {
    let mut cmd = Command::new(if have_gnu_time { GNU_TIME } else { program });
    if have_gnu_time {
        cmd.arg("-v").arg(program);
    }
    cmd.args(args);

    let start = Instant::now();
    let outcome = run_with_timeout(&mut cmd, timeout, io_dir)?;
    let wall = start.elapsed();

    // `status: None` means the poll loop killed it at the ceiling.
    if outcome.status.is_none() {
        return Ok(Err(SampleFailure {
            timed_out: true,
            stderr: outcome.stderr,
        }));
    }
    if !outcome.success() {
        return Ok(Err(SampleFailure {
            timed_out: false,
            stderr: outcome.stderr,
        }));
    }
    Ok(Ok(SubprocessSample {
        wall,
        peak_rss_bytes: have_gnu_time
            .then(|| parse_peak_rss(&outcome.stderr))
            .flatten(),
    }))
}

/// Measure `program args...` over `warmup` discarded + `iters` recorded
/// runs. Returns the recorded samples, or the first run's failure (a
/// failed compile is not a data point, but it is worth recording as one).
pub fn measure(
    program: &str,
    args: &[String],
    warmup: u32,
    iters: u32,
    timeout: Duration,
    io_dir: &Path,
    // Called before each run with the 1-based iteration and the total
    // (warmup + iters), so a caller can show live iteration progress.
    on_iter: &mut dyn FnMut(u32, u32),
) -> std::io::Result<Measurement> {
    let have_gnu_time = Path::new(GNU_TIME).exists();
    let total = warmup + iters;
    let mut samples = Vec::with_capacity(iters as usize);
    for iteration in 0..total {
        on_iter(iteration + 1, total);
        match run_once(program, args, timeout, io_dir, have_gnu_time)? {
            Err(failure) => return Ok(Measurement::Failed(failure)),
            Ok(sample) => {
                if iteration >= warmup {
                    samples.push(sample);
                }
            }
        }
    }
    Ok(Measurement::Measured(samples))
}

/// Peak-RSS median across samples in bytes, `None` if unmeasured. RSS is
/// all-or-nothing: if any sample lacks it, the whole config reports none.
pub fn median_peak_rss(samples: &[SubprocessSample]) -> Option<u64> {
    let values: Vec<u64> = samples.iter().filter_map(|s| s.peak_rss_bytes).collect();
    if values.len() != samples.len() {
        return None;
    }
    median(&values)
}

/// Pull "Maximum resident set size (kbytes): N" out of GNU time's
/// verbose report and return bytes.
fn parse_peak_rss(stderr: &str) -> Option<u64> {
    for line in stderr.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Maximum resident set size (kbytes):") {
            let kib: u64 = rest.trim().parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gnu_time_rss() {
        let sample = "\tCommand being timed: \"true\"\n\
                      \tMaximum resident set size (kbytes): 1360\n\
                      \tExit status: 0\n";
        assert_eq!(parse_peak_rss(sample), Some(1360 * 1024));
    }

    #[test]
    fn absent_rss_line_is_none() {
        assert_eq!(parse_peak_rss("no rss here\n"), None);
    }
}
