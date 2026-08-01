//! The optimisation pipeline: an ordered list of passes over the RVSDG,
//! each wrapped in uniform observability (a tracing span, wall time,
//! Rust-heap movement, graph shape before/after) plus the pass's own
//! effect counters. Reports are plain values returned to the caller;
//! nothing here prints or stores.
//!
//! Measurement split, deliberately: the DRIVER measures everything
//! observable from outside the pass (time, heap, shape), so a pass can
//! never misreport them; a pass reports only counters that are cheap
//! inside loops it already runs and would cost a graph walk to derive
//! afterwards (see [`dead_node_elimination::DneEffects`]).

use std::time::{Duration, Instant};

use color_eyre::eyre::eyre;
use serde::Serialize;

use crate::rvsdg::RVSDGMod;
use crate::stats::heap;

pub mod dead_node_elimination;

pub use dead_node_elimination::DneEffects;

#[derive(Debug)]
pub enum OptPass {
    DeadNodeElimination,
}

impl OptPass {
    pub fn name(&self) -> &'static str {
        match self {
            OptPass::DeadNodeElimination => "DeadNodeElimination",
        }
    }

    fn run_pass(&self, rvsdg_mod: &mut RVSDGMod) -> color_eyre::Result<PassEffects> {
        match self {
            OptPass::DeadNodeElimination => Ok(PassEffects::DeadNodeElimination(
                rvsdg_mod.opt_dead_node_elimination()?,
            )),
        }
    }
}

/// O(1) size facts read off the module around every pass. Deep shape
/// (kind distributions, slot metrics) stays in the census, which costs
/// a walk and is flag-gated; this is the always-on core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GraphShape {
    pub values: usize,
    pub regions: usize,
    pub value_pool_entries: usize,
    pub region_pool_entries: usize,
    pub u32_pool_entries: usize,
    pub match_arm_pool_entries: usize,
}

impl GraphShape {
    pub fn measure(module: &RVSDGMod) -> Self {
        let mut shape = Self {
            values: 0,
            regions: 0,
            value_pool_entries: 0,
            region_pool_entries: 0,
            u32_pool_entries: 0,
            match_arm_pool_entries: 0,
        };
        for graph in module.graphs.iter().flatten() {
            shape.values += graph.value_kinds.len();
            shape.regions += graph.regions.len();
            shape.value_pool_entries += graph.value_pool.len();
            shape.region_pool_entries += graph.region_pool.len();
            shape.u32_pool_entries += graph.u32_pool.len();
            shape.match_arm_pool_entries += graph.match_arm_pool.len();
        }
        shape
    }
}

/// Pass-reported counters. Exhaustive over passes, so adding a pass
/// forces a decision about what it reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PassEffects {
    DeadNodeElimination(DneEffects),
}

fn duration_as_ms<S: serde::Serializer>(
    duration: &Duration,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(duration.as_secs_f64() * 1000.0)
}

/// One pass execution: driver-measured core plus the pass's counters.
/// Serializes directly (durations as fractional milliseconds), so a
/// field added here reaches every consumer without a mirror to forget.
#[derive(Debug, Serialize)]
pub struct PassReport {
    pub pass: &'static str,
    #[serde(rename = "duration_ms", serialize_with = "duration_as_ms")]
    pub duration: Duration,
    /// Whole-graph verification after this pass (--verify-all only;
    /// zero otherwise). Kept out of `duration` so pass cost and
    /// checking cost never blur, but reported so no wall time vanishes.
    #[serde(rename = "verify_ms", serialize_with = "duration_as_ms")]
    pub verify_duration: Duration,
    /// Rust-heap live bytes around the pass; both zero when the binary
    /// does not install the counting allocator (see [`heap`]).
    pub heap_live_before_bytes: usize,
    pub heap_live_after_bytes: usize,
    pub shape_before: GraphShape,
    pub shape_after: GraphShape,
    pub effects: PassEffects,
}

impl std::fmt::Display for PassReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Signed: expansion passes (inlining, promotion) grow the graph.
        let delta = self.shape_after.values as i64 - self.shape_before.values as i64;
        let delta_pct = if self.shape_before.values > 0 {
            delta as f64 * 100.0 / self.shape_before.values as f64
        } else {
            0.0
        };
        write!(
            f,
            "{}: {:.1}ms, values {} -> {} ({delta_pct:+.1}%), regions {} -> {}",
            self.pass,
            self.duration.as_secs_f64() * 1000.0,
            self.shape_before.values,
            self.shape_after.values,
            self.shape_before.regions,
            self.shape_after.regions,
        )?;
        if self.heap_live_before_bytes > 0 {
            let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
            write!(
                f,
                ", heap {:.1}MiB -> {:.1}MiB",
                mib(self.heap_live_before_bytes),
                mib(self.heap_live_after_bytes),
            )?;
        }
        match &self.effects {
            PassEffects::DeadNodeElimination(effects) => write!(
                f,
                ", slots dropped: gamma inputs {}, theta loop vars {}, result entries {}; pinned projections {}",
                effects.gamma_input_slots_dropped,
                effects.theta_loop_var_slots_dropped,
                effects.result_entries_dropped,
                effects.pinned_projections,
            ),
        }
    }
}

/// What one pipeline run produced: the per-pass reports plus the
/// pipeline-level verification time that belongs to no single pass
/// (the --verify-all check BEFORE the first pass).
#[derive(Debug, Default)]
pub struct PipelineReport {
    pub passes: Vec<PassReport>,
    pub pre_verify_duration: Duration,
}

impl PipelineReport {
    /// Everything the pipeline spent verifying: the pre-pass check plus
    /// each pass's post-check. Zero unless --verify-all.
    pub fn total_verify_duration(&self) -> Duration {
        self.pre_verify_duration
            + self
                .passes
                .iter()
                .map(|report| report.verify_duration)
                .sum::<Duration>()
    }
}

impl RVSDGMod {
    /// Run `passes` in order, returning one report per pass. With
    /// `verify_all`, the whole graph is verified before the first pass
    /// (naming construction as the culprit rather than the pass that
    /// trips over it) and after every pass, so a broken invariant is
    /// attributed to the pass that broke it instead of surfacing as a
    /// miscompile downstream.
    pub fn optimise(
        &mut self,
        passes: Vec<OptPass>,
        verify_all: bool,
    ) -> color_eyre::Result<PipelineReport> {
        let mut report = PipelineReport::default();
        if verify_all {
            let started = Instant::now();
            self.verify_stage("before any passes")?;
            report.pre_verify_duration = started.elapsed();
        }
        report.passes.reserve(passes.len());
        for pass in passes {
            // Structural per-pass span: a pass shows up in --trace by
            // being run, not by remembering its own instrument
            // attribute. The shape fields make trace args readable
            // without the report values.
            let span = tracing::info_span!(
                "opt_pass",
                pass = pass.name(),
                values_before = tracing::field::Empty,
                values_after = tracing::field::Empty,
            );
            let shape_before = GraphShape::measure(self);
            let heap_live_before_bytes = heap::live_bytes();
            let started = Instant::now();
            let effects = {
                let _guard = span.enter();
                pass.run_pass(self)?
            };
            let mut pass_report = PassReport {
                pass: pass.name(),
                duration: started.elapsed(),
                verify_duration: Duration::ZERO,
                heap_live_before_bytes,
                heap_live_after_bytes: heap::live_bytes(),
                shape_before,
                shape_after: GraphShape::measure(self),
                effects,
            };
            span.record("values_before", pass_report.shape_before.values);
            span.record("values_after", pass_report.shape_after.values);
            if verify_all {
                let started = Instant::now();
                self.verify_stage(&format!("after {}", pass_report.pass))?;
                pass_report.verify_duration = started.elapsed();
            }
            report.passes.push(pass_report);
        }
        Ok(report)
    }

    pub fn optimise_default(&mut self, verify_all: bool) -> color_eyre::Result<PipelineReport> {
        let passes = vec![OptPass::DeadNodeElimination];
        self.optimise(passes, verify_all)
    }

    fn verify_stage(&self, stage: &str) -> color_eyre::Result<()> {
        let errs = self.verify();
        if errs.is_empty() {
            return Ok(());
        }
        let listed: Vec<String> = errs.iter().take(10).map(|e| format!("  {e}")).collect();
        Err(eyre!(
            "verification failed {stage}: {} errors\n{}{}",
            errs.len(),
            listed.join("\n"),
            if errs.len() > listed.len() {
                "\n  ..."
            } else {
                ""
            },
        ))
    }
}
