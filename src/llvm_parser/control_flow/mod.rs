//! **The CFG -> RVSDG control-flow construction pipeline** (Bahmann,
//! Reissmann, Jahre, Meyer 2015, section 4), in two phases:
//!
//! 1. [`build_overlay`] -- restructure the control flow into overlay
//!    records: the loop pass ([`loop_pass`]) collapses every strongly
//!    connected component into a single-entry, tail-controlled loop, then
//!    the branch pass ([`branch_pass`]) gives every branch exactly one
//!    continuation point. The records ([`overlay`]) are a complete
//!    description of the restructured graph; [`view`] is the traversal over
//!    it and [`partition`] the shared alternative-subgraph computation.
//! 2. [`emit`] -- walk the restructured graph and emit the RVSDG, with the
//!    scoped symbol table ([`scopes`]) observing each construct's inputs
//!    (captures) and outputs (writes), assembled into gamma/theta nodes
//!    afterwards, exactly the paper's BUILDRVSDG* order.

pub mod analysis;
pub mod branch_pass;
pub mod emit;
pub mod loop_pass;
pub mod oracle;
pub mod overlay;
pub mod partition;
pub mod scopes;
pub mod view;

/// Build the complete restructuring overlay for one function: the loop pass
/// (collapse every strongly connected component into a single-entry,
/// tail-controlled loop), then the branch pass (give every branch exactly
/// one continuation point). The result is a full description of the
/// restructured control flow graph; emission is then a mechanical walk.
/// `diverging[b]` marks blocks whose terminator has no successors.
pub(in crate::llvm_parser) fn build_overlay(
    mapper: &crate::llvm_parser::block_mapper::BasicBlockMapper,
    tree: &crate::llvm_parser::scc::SccTree,
    diverging: Vec<bool>,
) -> overlay::Overlay {
    let mut built = overlay::Overlay::new(mapper, diverging, tree.len());
    loop_pass::run_loop_pass(mapper, tree, &mut built);
    branch_pass::run_branch_pass(mapper, tree, &mut built);
    built
}
