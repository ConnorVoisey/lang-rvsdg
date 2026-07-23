//! Benchmark harness building blocks, used by the `compile_bench` binary.
//!
//! [`measure`] holds the generic, suite-agnostic primitives (timed
//! subprocess runs, medians, governor checks). Everything else composes
//! the compile-time benchmark: [`counters`] reads hardware counters,
//! [`compile_time`] runs the in-process pipeline and the config matrix,
//! [`subprocess`] measures whole-process wall / peak RSS, [`record`] is
//! the durable JSON schema, and [`suite`] is the preset registry that
//! turns a path into the programs it contains.

pub mod cachegrind;
pub mod compile_time;
pub mod counters;
pub mod measure;
pub mod record;
pub mod subprocess;
pub mod suite;
