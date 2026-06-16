//! Region and CFG analyses shared by both phases of the control-flow
//! construction pipeline:
//!
//! - [`branches`] -- continuation-point analysis (section 4.2) plus the small
//!   per-branch lowering helpers (switch control predicate, join-phi resolution);
//! - [`loops`] -- the irreducible-loop dispatch-dominator table (section 4.1);
//! - [`signature`] -- phi-driven region live-ins, read directly off the LLVM phi
//!   nodes.

pub mod branches;
pub mod loops;
pub mod signature;
