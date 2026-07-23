//! In-process hardware counters for phase-level compile-time
//! measurement, via `perf_event_open` (Linux). A phase is measured by
//! enabling a counter group around a closure that runs the phase
//! directly -- no subprocess, so the counts cover exactly the RVSDG
//! work and nothing else (no process startup, no frontend).
//!
//! Graceful degradation is mandatory: `perf_event_paranoid` or a
//! missing `CAP_PERFMON` makes counter creation fail, and a benchmark
//! that panicked there would be useless on most machines. When the
//! group cannot be built the counters read `None` and wall-clock (which
//! always works) carries the measurement, with a one-line warning from
//! [`availability_warning`].
//!
//! The four events are chosen for what a cache-locality change actually
//! moves: `cycles` and `cache_misses` are the signal; `instructions` is
//! near-deterministic and the stable sentinel for later algorithmic
//! passes; `cache_references` gives a miss rate.

use std::time::{Duration, Instant};

use perf_event::events::Hardware;
use perf_event::{Builder, Counter, Group};

/// One phase's measurement: always a wall time, plus counters when the
/// kernel allowed them.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseMetrics {
    pub wall: Duration,
    pub cycles: Option<u64>,
    pub instructions: Option<u64>,
    pub cache_misses: Option<u64>,
    pub cache_references: Option<u64>,
}

/// A reusable counter group. Built once and reset around each phase, so
/// the `perf_event_open` syscalls happen at setup, not per measurement.
/// `None` inside means counters are unavailable; wall-clock still works.
#[derive(Debug)]
pub struct Counters {
    inner: Option<CounterGroup>,
}

#[derive(Debug)]
struct CounterGroup {
    group: Group,
    cycles: Counter,
    instructions: Counter,
    cache_misses: Counter,
    cache_references: Counter,
}

impl Counters {
    /// Build the counter group, or fall back to wall-clock only. Never
    /// errors: a benchmark must run on a machine without counter access.
    pub fn new() -> Self {
        Counters {
            inner: Self::try_build().ok(),
        }
    }

    fn try_build() -> std::io::Result<CounterGroup> {
        let mut group = Group::new()?;
        // All four in one group so they are scheduled together and their
        // counts cover the same window; four hardware events fit the
        // general-purpose PMU slots on any modern x86.
        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()?;
        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()?;
        let cache_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()?;
        let cache_references = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()?;
        Ok(CounterGroup {
            group,
            cycles,
            instructions,
            cache_misses,
            cache_references,
        })
    }

    fn counters_available(&self) -> bool {
        self.inner.is_some()
    }

    /// Run `phase` once, returning its result and metrics. Wall time is a
    /// plain `Instant` (cross-checks against a shell `time`); counters,
    /// when present, are reset-enabled-disabled-read tightly around the
    /// call so only the phase's work is counted.
    pub fn measure<R>(&mut self, phase: impl FnOnce() -> R) -> (R, PhaseMetrics) {
        match &mut self.inner {
            None => {
                let start = Instant::now();
                let result = phase();
                let wall = start.elapsed();
                (
                    result,
                    PhaseMetrics {
                        wall,
                        ..Default::default()
                    },
                )
            }
            Some(group) => {
                // Reset so counts are per-phase, not cumulative. If any
                // syscall here fails mid-run, drop to wall-only for this
                // phase rather than abort the benchmark.
                let armed = group
                    .group
                    .reset()
                    .and_then(|_| group.group.enable())
                    .is_ok();
                let start = Instant::now();
                let result = phase();
                let wall = start.elapsed();
                let counts = if armed {
                    group.group.disable().ok();
                    group.group.read().ok()
                } else {
                    None
                };
                let metrics = match counts {
                    Some(counts) => PhaseMetrics {
                        wall,
                        cycles: Some(counts[&group.cycles]),
                        instructions: Some(counts[&group.instructions]),
                        cache_misses: Some(counts[&group.cache_misses]),
                        cache_references: Some(counts[&group.cache_references]),
                    },
                    None => PhaseMetrics {
                        wall,
                        ..Default::default()
                    },
                };
                (result, metrics)
            }
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

/// A one-line note when counters are unavailable, so a wall-clock-only
/// run says why rather than silently omitting the columns.
pub fn availability_warning(counters: &Counters) -> Option<String> {
    (!counters.counters_available()).then(|| {
        let paranoid = std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "?".to_string());
        format!(
            "hardware counters unavailable (perf_event_paranoid={paranoid}); wall-clock only. \
             For counters: sudo sysctl kernel.perf_event_paranoid=1"
        )
    })
}
