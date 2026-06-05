//! Test scaffolding shared across the region module's test mods.
//!
//! Tests in `loops.rs`, `branches.rs`, and `phi.rs` all need to parse a
//! small LLVM IR snippet, build dominator and SCC analyses on it, and
//! spin up a `RegionLowerer` to run assertions against. `TestFn` owns
//! everything those steps need; `scc_for` is a thin helper to look up
//! a Phase-1-shaped strongly connected component by header name; the
//! `local_name` helper constructs the LLVM IR's textual local-name form.

use llvm_ir::{Module, Name};
use rustc_hash::FxHashMap;
use std::sync::Mutex;

use crate::{
    llvm_parser::{
        FnCtx,
        block_mapper::{BasicBlockId, BasicBlockMapper},
        dominance::{ForwardView, ReverseView, compute_dominance},
        instructions::RegionLowerer,
        scc_tree::{SccTree, SccTreeNodeId},
    },
    rvsdg::{Linkage, RVSDGMod, ValueId, builder::RegionBuilder},
};

/// The llvm-ir parser's first call lazily initialises an attribute-kind
/// table that asserts `nocapture` is present. LLVM 19 removed
/// `nocapture`, so that init panics if it races against itself.
/// Serialise parse calls until we move off llvm-ir 0.11.3.
static LLVM_PARSE_LOCK: Mutex<()> = Mutex::new(());

/// Owns everything a `FnCtx` borrows from. Tests build one of these
/// and either borrow a `FnCtx` directly (for free-function helpers
/// like `phi_instructions_at`) or call `with_lowerer` to spin up a
/// real `RegionLowerer` and run assertions against its methods.
pub(super) struct TestFn {
    pub(super) module: Module,
    pub(super) bb_mapper: BasicBlockMapper,
    pub(super) scc_tree: SccTree,
    pub(super) scc_entry_block_to_id: Vec<Option<SccTreeNodeId>>,
    pub(super) immediate_dominators: Vec<Option<BasicBlockId>>,
    pub(super) post_immediate_dominators: Vec<Option<BasicBlockId>>,
    pub(super) exit_block_id: BasicBlockId,
}

impl TestFn {
    pub(super) fn from_ir(ir: &str) -> Self {
        let module = {
            let _guard = LLVM_PARSE_LOCK.lock().unwrap();
            Module::from_ir_str(ir).expect("parse test IR")
        };
        assert!(
            !module.functions.is_empty(),
            "test IR must define at least one function"
        );
        let func = &module.functions[0];
        let mut bb_mapper = BasicBlockMapper::new(func.basic_blocks.len());
        for block in &func.basic_blocks {
            bb_mapper.intern(&block.name);
        }
        let exit_name = bb_mapper.exit_name();
        let exit_block_id = bb_mapper.intern(&exit_name);
        for (i, block) in func.basic_blocks.iter().enumerate() {
            let from = BasicBlockId(i as u32);
            match &block.term {
                llvm_ir::Terminator::Br(br) => {
                    let to = *bb_mapper.get_expect(&br.dest);
                    bb_mapper.add_connection(from, to);
                }
                llvm_ir::Terminator::CondBr(cb) => {
                    let t = *bb_mapper.get_expect(&cb.true_dest);
                    let f = *bb_mapper.get_expect(&cb.false_dest);
                    bb_mapper.add_connection(from, t);
                    bb_mapper.add_connection(from, f);
                }
                llvm_ir::Terminator::Ret(_) => {
                    bb_mapper.add_connection(from, exit_block_id);
                }
                llvm_ir::Terminator::Switch(sw) => {
                    let d = *bb_mapper.get_expect(&sw.default_dest);
                    bb_mapper.add_connection(from, d);
                    for (_, dest) in &sw.dests {
                        let did = *bb_mapper.get_expect(dest);
                        bb_mapper.add_connection(from, did);
                    }
                }
                _ => {}
            }
        }
        let immediate_dominators = compute_dominance(&ForwardView {
            nodes: &bb_mapper.blocks,
            entry: BasicBlockId(0),
        });
        let post_immediate_dominators = compute_dominance(&ReverseView {
            nodes: &bb_mapper.blocks,
            exit: exit_block_id,
        });
        let scc_tree = SccTree::build(&bb_mapper);
        let scc_entry_block_to_id = scc_tree.entry_block_to_node(bb_mapper.blocks.len());
        Self {
            module,
            bb_mapper,
            scc_tree,
            scc_entry_block_to_id,
            immediate_dominators,
            post_immediate_dominators,
            exit_block_id,
        }
    }

    pub(super) fn fn_ctx(&self) -> FnCtx<'_> {
        FnCtx {
            llvm_mod: &self.module,
            func: &self.module.functions[0],
            bb_mapper: &self.bb_mapper,
            scc_tree: &self.scc_tree,
            scc_entry_block_to_id: &self.scc_entry_block_to_id,
            immediate_dominators: &self.immediate_dominators,
            post_immediate_dominators: &self.post_immediate_dominators,
            exit_block_id: self.exit_block_id,
        }
    }

    pub(super) fn block(&self, name: &str) -> BasicBlockId {
        *self
            .bb_mapper
            .get(&local_name(name))
            .unwrap_or_else(|| panic!("test IR has no block named %{name}"))
    }

    /// Spin up an `RVSDGMod` + `RegionBuilder` + `RegionLowerer` and run
    /// `f` against the lowerer. `name_to_value` seeds the lowerer's
    /// outer SSA map (stand-in for values defined before the
    /// gamma/theta node we'd be lowering). The closure runs the
    /// assertions; nothing about the RVSDG it builds is observed, so we
    /// don't bother finishing the function definition.
    pub(super) fn with_lowerer<R>(
        &self,
        name_to_value: FxHashMap<Name, ValueId>,
        f: impl FnOnce(&mut RegionLowerer<'_, '_, '_>) -> R,
    ) -> R {
        let mut rvsdg = RVSDGMod::new_host("test".to_string());
        let fn_id = rvsdg.declare_fn("test".to_string(), &[], &[], Linkage::Internal);
        let mut rb = RegionBuilder::new_from_func(&mut rvsdg, fn_id);
        let fn_ctx = self.fn_ctx();
        let mut lowerer = RegionLowerer {
            rb: &mut rb,
            name_to_value,
            fn_ctx: &fn_ctx,
        };
        f(&mut lowerer)
    }
}

pub(super) fn local_name(s: &str) -> Name {
    Name::Name(Box::new(s.to_string()))
}

/// Resolved shape of a Phase-1-supported strongly connected component:
/// single entry vertex, single exit arc, single repetition arc. The
/// four fields share a type (three `BasicBlockId`s plus the component
/// id), so a named struct avoids the easy positional mix-up that an
/// unnamed tuple would invite at call sites.
pub(super) struct SccTopology {
    pub(super) id: SccTreeNodeId,
    pub(super) header: BasicBlockId,
    pub(super) latch: BasicBlockId,
    /// The single exit arc as `(src, dst)`. Tests asserting on the
    /// destination use `exit_arc.1`; tests wiring `analyze_loop` pass
    /// the full pair.
    pub(super) exit_arc: (BasicBlockId, BasicBlockId),
}

/// Look up the strongly connected component whose entry vertex is
/// `entry_block_name`. Assumes the caller's fixture has a single-entry
/// single-exit single-back-edge shape, which is the Phase 1 supported
/// case. Mirrors what `RegionLowerer::loop_at` plus
/// `lower_scc_as_theta` resolve at the start of `lower_region`'s loop
/// dispatch.
pub(super) fn scc_for(test_fn: &TestFn, entry_block_name: &str) -> SccTopology {
    let entry = test_fn.block(entry_block_name);
    let id = test_fn.scc_entry_block_to_id[entry.0 as usize]
        .unwrap_or_else(|| panic!("block %{entry_block_name} is not an SCC entry vertex"));
    let arcs = &test_fn.scc_tree.arcs[id.0 as usize];
    assert_eq!(
        1,
        arcs.entry_blocks.len(),
        "scc_for assumes single-entry component"
    );
    assert_eq!(
        1,
        arcs.exit_arcs.len(),
        "scc_for assumes single-exit component"
    );
    assert_eq!(
        1,
        arcs.repetition_arcs.len(),
        "scc_for assumes single-back-edge component"
    );
    SccTopology {
        id,
        header: arcs.entry_blocks[0],
        latch: arcs.repetition_arcs[0].0,
        exit_arc: arcs.exit_arcs[0],
    }
}
