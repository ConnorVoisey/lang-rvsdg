//! Shared scaffolding for unit tests across `dominance`, `loop_arcs`, and
//! `loops`. Three independent copies of `init` existed before this module;
//! consolidate so future additions don't add a fourth.

#![cfg(test)]

use llvm_ir::Name;

use crate::llvm_parser::{
    block_mapper::{BasicBlockId, BasicBlockMapper},
    dominance::{ForwardView, ReverseView, compute_dominance},
    strongly_connected_components::Scc,
};

/// Build a `BasicBlockMapper` with `n` blocks (named `%0` .. `%n-1`) and no
/// edges. Returns the mapper plus a Vec of the assigned `BasicBlockId`s in
/// the same order. Test callers add edges via `mapper.add_connection`.
pub(crate) fn init(n: usize) -> (BasicBlockMapper, Vec<BasicBlockId>) {
    let mut mapper = BasicBlockMapper::new(n);
    let bbs = (0..n)
        .map(|i| mapper.intern(&Name::Number(i)))
        .collect::<Vec<_>>();
    (mapper, bbs)
}

/// Build a non-trivial `Scc` from an explicit block list. Useful for
/// `loop_arcs` tests that want to bypass Tarjan and assert on a known
/// SCC shape directly.
pub(crate) fn scc_of(blocks: &[BasicBlockId]) -> Scc {
    Scc {
        blocks: blocks.iter().copied().collect(),
        is_trivial: false,
    }
}

/// Compute forward immediate dominators for the function described by
/// `mapper`. Assumes the entry block is `BasicBlockId(0)`.
pub(crate) fn idoms_of(mapper: &BasicBlockMapper) -> Vec<Option<BasicBlockId>> {
    let view = ForwardView {
        nodes: &mapper.blocks,
        entry: BasicBlockId(0),
    };
    compute_dominance(&view)
}

/// Compute post-dominators (reverse dominance) for the function described
/// by `mapper`, treating `exit` as the synthetic exit block.
pub(crate) fn post_idoms_of(
    mapper: &BasicBlockMapper,
    exit: BasicBlockId,
) -> Vec<Option<BasicBlockId>> {
    let view = ReverseView {
        nodes: &mapper.blocks,
        exit,
    };
    compute_dominance(&view)
}
