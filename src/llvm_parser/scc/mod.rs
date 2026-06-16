//! Loop-structure analysis: discover the function's loops as strongly connected
//! components and describe their nesting and arc structure. This is the input
//! the restructuring transform (phase 1 of [`super::control_flow`]) consumes to
//! turn each loop into a theta node.
//!
//! - [`components`] -- Tarjan's strongly-connected-component algorithm; a
//!   non-trivial component (a cycle) is a loop.
//! - [`arcs`] -- per-component entry / exit / repetition arc classification
//!   (Bahmann et al. 2015 section 4.1).
//! - [`tree`] -- the SCC *nesting* tree ([`SccTree`]): the artifact the pipeline
//!   reads (via `FnCtx`), built by recursively re-running Tarjan inside each
//!   component with its repetition arcs removed, with [`arcs`] attached per node.

pub mod arcs;
pub mod components;
pub mod tree;

pub use tree::{SccTree, SccTreeNodeId};
