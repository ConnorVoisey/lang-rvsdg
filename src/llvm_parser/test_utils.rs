//! Shared scaffolding for the CFG-analysis unit tests (the `scc` modules and
//! the `control_flow` passes). Several independent copies of `init` existed
//! before this module; consolidate so future additions don't add another.

#![cfg(test)]

use llvm_ir::Name;

use crate::llvm_parser::block_mapper::{BasicBlockId, BasicBlockMapper};

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
