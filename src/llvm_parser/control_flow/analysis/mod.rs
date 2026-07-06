//! Small shared CFG helpers for the control-flow pipeline: [`signature`]
//! holds the phi lookups (leading phi run of a block, incoming value for a
//! given predecessor) used by arc-payload application and the emitter.

pub mod signature;
