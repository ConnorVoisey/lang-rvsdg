//! The **Structured Region Tree (RST)**: the data structure the two phases of
//! the control-flow pipeline pass between each other. Phase 1
//! ([`super::restructure`]) builds it from the LLVM control flow graph; phase 2
//! ([`super::construct`]) walks it and emits the RVSDG. It describes *control
//! structure only* -- every leaf references an original [`BasicBlockId`]; no
//! instruction is copied.
//!
//! A region's *role* (where it sits in an enclosing construct) fixes which exits
//! are valid: a sequential region may return, a loop body may produce a mixed
//! demux, a capture arm may only route, and so on. Each role has its own exit
//! enum so the construction walk handles exactly the shapes its role permits --
//! invalid combinations are unrepresentable rather than runtime errors.

use crate::llvm_parser::{block_mapper::BasicBlockId, scc::SccTreeNodeId};

/// A structured region: a straight-line sequence of items followed by how control
/// leaves the region (the role-specific `exit`). Lowered top to bottom, threading
/// the RVSDG state edge. The item list is identical across roles; only the exit
/// kind differs, so the region is generic over its exit type.
#[derive(Debug)]
pub(in crate::llvm_parser) struct StructuredRegion<X> {
    pub items: Vec<RegionItem>,
    pub exit: X,
}

/// A top-level body / gamma split-arm / demux-tail region. May reach a
/// continuation, return, diverge, or end in an all-arms-return gamma.
pub(in crate::llvm_parser) type SeqRegion = StructuredRegion<SeqExit>;
/// A continuation-demux head arm. May reach a continuation, diverge, or route on.
pub(in crate::llvm_parser) type CaptureRegion = StructuredRegion<CaptureExit>;
/// An irreducible-loop entry region. May reach an entry vertex or route on.
pub(in crate::llvm_parser) type EntryRegion = StructuredRegion<EntryExit>;
/// A loop body. May reach a loop boundary, route, or end in a mixed demux.
pub(in crate::llvm_parser) type LoopBodyRegion = StructuredRegion<LoopBodyExit>;
/// A loop-body-demux head arm. May reach a continuation or route on.
pub(in crate::llvm_parser) type LoopCaptureRegion = StructuredRegion<LoopCaptureExit>;

/// One structured item within a region. Shared by every region role; the
/// sub-regions a gamma/theta contains have fixed roles (a gamma's split arms and
/// a demux gamma's tails are [`SeqRegion`]s, a demux gamma's head arms are
/// [`CaptureRegion`]s, a theta body is a [`LoopBodyRegion`], an irreducible
/// loop's entry region is an [`EntryRegion`]).
#[derive(Debug)]
pub(in crate::llvm_parser) enum RegionItem {
    /// Lower this block's straight-line (non-phi, non-terminator) instructions in
    /// place. The block's phi destinations are bound from the preceding item's
    /// block (the linear predecessor).
    Block(BasicBlockId),
    /// A branch (gamma) construct.
    Gamma(GammaNode),
    /// A loop (theta) construct.
    Theta(ThetaNode),
}

/// A loop construct lowered to a theta node. `body` is the loop body structured
/// with the loop boundary `{header} ∪ exit_blocks` (reaching `header` is a
/// repeat, reaching an exit vertex is an exit); its leaves are the per-iteration
/// loop-variable vector.
#[derive(Debug)]
pub(in crate::llvm_parser) struct ThetaNode {
    pub scc: SccTreeNodeId,
    /// Distinct exit targets, in exit `q` order; the index is the exit `q` value
    /// for a multi-exit loop.
    pub exit_blocks: Vec<BasicBlockId>,
    /// For a multi-exit loop, the post-theta demux on the exit `q`: `None` for a
    /// single-exit loop (control resumes directly at the sole exit target).
    pub exit_demux: Option<ExitDemux>,
    pub kind: ThetaKind,
}

/// The two loop shapes a theta lowers.
#[derive(Debug)]
pub(in crate::llvm_parser) enum ThetaKind {
    /// A single-entry loop: one `body` region (boundary `{header} ∪ exit_blocks`).
    Reducible {
        header: BasicBlockId,
        body: LoopBodyRegion,
    },
    /// A multi-entry (irreducible) loop: an `entry_region` (from the dispatch
    /// dominator to the entry vertices, producing the entry `q` plus the entry-phi
    /// inits) and one body region per entry vertex (boundary `entries ∪
    /// exit_blocks`), dispatched on the entry `q` inside the theta.
    MultiEntry {
        entries: Vec<BasicBlockId>,
        entry_region: EntryRegion,
        bodies: Vec<LoopBodyRegion>,
    },
}

/// The post-theta exit demux of a multi-exit loop: a gamma on the exit `q`
/// selects which exit vertex was taken; `tails[i]` lowers `exit_blocks[i]`. How
/// the tails leave the dispatch is fixed by `merge`.
#[derive(Debug)]
pub(in crate::llvm_parser) struct ExitDemux {
    /// One tail per exit vertex, keyed by exit `q` (index = exit-block index).
    pub tails: Vec<SeqRegion>,
    pub merge: ExitMerge,
}

/// How a multi-exit loop's exit tails collectively leave the post-theta dispatch.
#[derive(Debug)]
pub(in crate::llvm_parser) enum ExitMerge {
    /// Every tail reconverges at a single continuation `join`; `tails[i]` lowers
    /// `exit_blocks[i]` to it and the enclosing region resumes at `join`. The
    /// theta is a mid-region item.
    Reconverge { join: BasicBlockId },
    /// No shared reconvergence: every tail returns or diverges. The dispatch is a
    /// return gamma on the exit `q` and the enclosing region returns the merged
    /// value, so the loop is the region's terminal (see [`SeqExit::LoopReturn`]).
    Return,
}

/// How a [`SeqRegion`] leaves: reach a continuation, return, diverge, or merge
/// per-arm returns in an all-arms-return gamma.
#[derive(Debug)]
pub(in crate::llvm_parser) enum SeqExit {
    /// Control reaches `reached` -- an enclosing construct's continuation/boundary
    /// block. `via` is the block control arrived from, used to resolve `reached`'s
    /// phis; `None` when they are already bound by a preceding gamma.
    ToContinuation {
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
    },
    /// The region's last block ends in `ret`; the return operand is read from that
    /// block's terminator.
    Return { block: BasicBlockId },
    /// The region diverges (`unreachable`, or a non-returning tail).
    Diverge,
    /// The region ends in a branch (terminating `head`) every arm of which returns
    /// or diverges. A gamma merges the per-arm return values and the function
    /// returns the gamma output.
    ReturnGamma {
        head: BasicBlockId,
        arms: Vec<SeqRegion>,
    },
    /// The region's terminal is a multi-exit loop whose exit tails do not
    /// reconverge (every one returns or diverges). The `theta` produces the exit
    /// `q`; a gamma on it dispatches to each tail and the function returns the
    /// merged value. `theta.exit_demux` is `Some` with [`ExitMerge::Return`].
    LoopReturn { theta: ThetaNode },
}

/// How a [`CaptureRegion`] (continuation-demux head arm) leaves.
#[derive(Debug)]
pub(in crate::llvm_parser) enum CaptureExit {
    /// Reaches a demux target `reached` (arriving from `via`).
    ToContinuation {
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
    },
    /// The arm diverges before reaching any demux target.
    Diverge,
    /// The arm ends in a branch (terminating `head`) every continuation of which
    /// is a demux target: a nested router whose arms recurse as capture arms.
    Route {
        head: BasicBlockId,
        arms: Vec<CaptureRegion>,
    },
}

/// How an [`EntryRegion`] (irreducible-loop entry region) leaves.
#[derive(Debug)]
pub(in crate::llvm_parser) enum EntryExit {
    /// Reaches an entry vertex `reached` (arriving from `via`).
    ToContinuation {
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
    },
    /// Ends in a branch (terminating `head`) every continuation of which is an
    /// entry vertex: a router whose arms recurse as entry regions.
    Route {
        head: BasicBlockId,
        arms: Vec<EntryRegion>,
    },
}

/// How a [`LoopBodyRegion`] leaves.
#[derive(Debug)]
pub(in crate::llvm_parser) enum LoopBodyExit {
    /// Reaches a loop boundary `reached` (a repeat vertex or exit vertex),
    /// arriving from `via`.
    ToContinuation {
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
    },
    /// Ends in a branch (terminating `head`) every continuation of which is a loop
    /// boundary: a router merging each arm's per-iteration leaf.
    Route {
        head: BasicBlockId,
        arms: Vec<LoopBodyRegion>,
    },
    /// Ends in a branch (terminating `head`) whose continuation points are a *mix*
    /// of in-body continuations (lowered here, once) and loop boundaries (routed
    /// to). `head` selects, each `targets[i]` is lowered once (in-region) or
    /// produces a boundary leaf; `arms` are the head arms (each a capture arm).
    Demux {
        head: BasicBlockId,
        arms: Vec<LoopCaptureRegion>,
        targets: Vec<DemuxBranchTarget>,
    },
}

/// How a [`LoopCaptureRegion`] (loop-body-demux head arm) leaves.
#[derive(Debug)]
pub(in crate::llvm_parser) enum LoopCaptureExit {
    /// Reaches a loop-demux continuation `reached` (arriving from `via`).
    ToContinuation {
        reached: BasicBlockId,
        via: Option<BasicBlockId>,
    },
    /// A nested router whose arms recurse as loop capture arms.
    Route {
        head: BasicBlockId,
        arms: Vec<LoopCaptureRegion>,
    },
}

/// One continuation of a [`LoopBodyExit::Demux`].
#[derive(Debug)]
pub(in crate::llvm_parser) struct DemuxBranchTarget {
    pub block: BasicBlockId,
    /// `Some(tail)` if `block` is an in-region continuation lowered here; `None`
    /// if it is an enclosing boundary (loop boundary / outer continuation).
    pub in_region_tail: Option<LoopBodyRegion>,
}

/// A branch construct: the head block whose terminator is the branch and how the
/// arms reconverge. The control predicate is rebuilt from `head`'s terminator.
#[derive(Debug)]
pub(in crate::llvm_parser) struct GammaNode {
    pub head: BasicBlockId,
    pub merge: GammaMerge,
}

/// How a gamma's arms reconverge.
#[derive(Debug)]
pub(in crate::llvm_parser) enum GammaMerge {
    /// All arms reconverge at a single continuation `join`; the gamma's outputs
    /// are `join`'s phis and the enclosing region resumes at `join` (or, if `join`
    /// is a boundary block, exits there). `arms` are the split arms, in terminator
    /// order.
    SingleJoin {
        join: BasicBlockId,
        arms: Vec<SeqRegion>,
    },
    /// The arms reconverge at more than one continuation: the `p`-demux. `head_arms`
    /// produce the demux predicate index plus each target's captured phis; the
    /// [`DemuxSpec`] lowers each target's tail exactly once.
    Demux {
        head_arms: Vec<CaptureRegion>,
        spec: DemuxSpec,
    },
}

/// The continuation-demux of a multi-continuation branch (Bahmann et al. section
/// 4.2). `demux_targets` are the continuation points plus the final `join`, in
/// ascending block order; `tails[i]` lowers `demux_targets[i]`'s continuation.
#[derive(Debug)]
pub(in crate::llvm_parser) struct DemuxSpec {
    pub demux_targets: Vec<BasicBlockId>,
    pub join: BasicBlockId,
    pub tails: Vec<DemuxTail>,
}

/// One demux target's continuation. The final `join` target is resolved in place
/// (its phis bound from each arm's exit edge), so it carries no region; every
/// other target carries the region lowering its continuation to `join`.
#[derive(Debug)]
pub(in crate::llvm_parser) enum DemuxTail {
    /// The reconvergence block itself.
    Join,
    /// A continuation point's tail region, lowered to `join` exactly once.
    Tail(SeqRegion),
}
