//! Per-function precomputation for irreducible (multi-entry) loops.
//!
//! An irreducible loop is entered at more than one vertex -- several blocks
//! outside the loop branch into different points of the cycle, so there is no
//! single header. To lower it to one loop node, every entry is funnelled through
//! one synthetic header that, on entry and on each repeat, resumes at the right
//! vertex according to a selector value (called the entry `q`). That selector
//! has to be chosen before control enters the loop, at the block that dominates
//! all the entry vertices -- the loop's *dispatch dominator*. This module
//! precomputes, per block, which loop (if any) it is the dispatch dominator of,
//! so the restructuring transform can spot the dispatch point in one lookup.

use crate::llvm_parser::{
    block_mapper::BasicBlockId,
    dominance::dominates,
    scc::{SccTree, SccTreeNodeId},
};

/// Per-block lookup: for each block, the multi-entry (irreducible) loop whose
/// **dispatch dominator** that block is, or `None` if it is no loop's dispatch
/// dominator. The dispatch dominator is the nearest block dominating all of a
/// loop's entry vertices -- see the module docs for what it is used for.
pub(in crate::llvm_parser) fn compute_multi_entry_dispatch(
    scc_tree: &SccTree,
    idoms: &[Option<BasicBlockId>],
    block_count: usize,
) -> Vec<Option<SccTreeNodeId>> {
    let mut dispatch = vec![None; block_count];
    for (node_index, arcs) in scc_tree.arcs.iter().enumerate() {
        let entries = &arcs.entry_blocks;
        if entries.len() <= 1 {
            continue;
        }
        if let Some(dispatch_dom) = common_dom_of_entries(entries, idoms) {
            dispatch[dispatch_dom.0 as usize] = Some(SccTreeNodeId(node_index as u32));
        }
    }
    dispatch
}

/// The nearest common dominator of `entries` (walking up the idom tree from the
/// first entry until it dominates all), or `None` if there is none.
fn common_dom_of_entries(
    entries: &[BasicBlockId],
    idoms: &[Option<BasicBlockId>],
) -> Option<BasicBlockId> {
    let mut candidate = *entries.first()?;
    loop {
        if entries
            .iter()
            .all(|&entry| dominates(candidate, entry, idoms))
        {
            return Some(candidate);
        }
        // Climb to the immediate dominator. The function entry's idom is itself,
        // so a self-parent means we reached the root without a common dominator.
        match idoms[candidate.0 as usize] {
            Some(parent) if parent != candidate => candidate = parent,
            _ => return None,
        }
    }
}
