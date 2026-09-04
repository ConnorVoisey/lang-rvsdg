use crate::rvsdg::{
    RegionId, ValueId, ValuePool,
    function_graph::FunctionGraph,
    state::{State, StateGroup},
};

#[derive(Debug, Clone)]
pub struct Region {
    /// The gamma/theta value this region belongs to. The graph only
    /// stores the forward direction during emission (the construct value
    /// does not exist until its regions are finished), so it is created
    /// as [`ValueId::INVALID`] and stamped by the construct's finaliser.
    /// Region 0 is the function body and stays owner-less; the verifier
    /// enforces both directions.
    pub owner: ValueId,
    /// Start of this region's INTERFACE BLOCK in the value pool, four
    /// contiguous segments written once at seal:
    /// [value params | state params | value results | state results].
    /// [`Region::UNSEALED`] until then -- an open region's growing lists
    /// live in the graph's construction scratch and every consumer reads
    /// them through the `region_params`/`region_results`/`region_nodes`
    /// accessors, never these fields directly.
    ///
    /// Parameters are appended on demand during emission (the emitter
    /// captures outer values into a region while its body is being
    /// built), so parameter VALUES interleave with body values in the
    /// value arrays; consumers identify a parameter by its position in
    /// the params segment, never by value-id arithmetic.
    pub interface_start: u32,
    /// Value params only; the state tail is counted separately so the
    /// value/state boundary is explicit, never inferred by type-scanning.
    pub params_len: u16,
    /// State params, one entry per chain in [memory, io] order: the
    /// parent-side values this region's chains start from. Slots are
    /// state-typed values; which chain a slot names is read off its
    /// type.
    pub state_params_len: u16,
    /// Value results only; see `params_len`.
    pub results_len: u16,
    /// State results, [memory, io]: each chain's value at region exit
    /// (the entry passed through when the region leaves the chain
    /// untouched; trailing pending reads flatten into the memory slot).
    pub state_results_len: u16,
    /// This region's nodes in topological (emission) order, as a span in
    /// the value pool, written at seal. Node IDS need not be ascending
    /// (passes append replacement values at high ids); the list order is
    /// the truth for emission and state order.
    pub nodes_start: u32,
    pub nodes_len: u32,
}

impl Region {
    /// Sentinel for a region whose interface block has not been sealed
    /// yet. Accessors panic on a pool read of an unsealed region instead
    /// of slicing from a bogus offset.
    pub const UNSEALED: u32 = u32::MAX;

    /// A freshly created, open region: owner and exit state stamped by
    /// the finaliser, lists living in construction scratch until seal.
    /// Crate-private so `FunctionGraph::create_region` (which registers
    /// the scratch) stays the single way a region comes to exist.
    pub(crate) fn new_open() -> Self {
        Region {
            owner: ValueId::INVALID,
            interface_start: Region::UNSEALED,
            params_len: 0,
            state_params_len: 0,
            results_len: 0,
            state_results_len: 0,
            nodes_start: 0,
            nodes_len: 0,
        }
    }

    pub fn is_sealed(&self) -> bool {
        self.interface_start != Region::UNSEALED
    }

    // -- Sealed interface-block segments --------------------------------
    //
    // Together with `write_blocks` these are the single definition of
    // the sealed layout [value params | state params | value results |
    // state results]; every consumer (graph accessors, purity, passes,
    // tests) reads segments through them rather than re-deriving
    // offsets.

    #[inline]
    fn state_params_start(&self) -> u32 {
        self.interface_start + self.params_len as u32
    }

    #[inline]
    fn results_start(&self) -> u32 {
        self.state_params_start() + self.state_params_len as u32
    }

    #[inline]
    fn state_results_start(&self) -> u32 {
        self.results_start() + self.results_len as u32
    }

    #[inline]
    pub(crate) fn params<'p>(&self, pool: &'p ValuePool) -> &'p [ValueId] {
        pool.slice(self.interface_start, self.params_len as usize)
    }

    #[inline]
    pub(crate) fn state_params<'p>(&self, pool: &'p ValuePool) -> &'p [ValueId] {
        pool.slice(self.state_params_start(), self.state_params_len as usize)
    }

    #[inline]
    pub(crate) fn results<'p>(&self, pool: &'p ValuePool) -> &'p [ValueId] {
        pool.slice(self.results_start(), self.results_len as usize)
    }

    #[inline]
    pub(crate) fn state_results<'p>(&self, pool: &'p ValuePool) -> &'p [ValueId] {
        pool.slice(self.state_results_start(), self.state_results_len as usize)
    }

    #[inline]
    pub(crate) fn state_params_mut<'p>(&self, pool: &'p mut ValuePool) -> &'p mut [ValueId] {
        pool.slice_mut(self.state_params_start(), self.state_params_len as usize)
    }

    #[inline]
    pub(crate) fn state_results_mut<'p>(&self, pool: &'p mut ValuePool) -> &'p mut [ValueId] {
        pool.slice_mut(self.state_results_start(), self.state_results_len as usize)
    }

    /// A region is pure when every state slot passes through: each
    /// chain's exit is its own entry, so executing the region has no
    /// observable effect. Sealed regions only (open regions read their
    /// state through scratch, and purity is a post-construction
    /// question).
    pub fn is_pure(&self, graph: &FunctionGraph) -> bool {
        debug_assert!(self.is_sealed());
        self.state_params(&graph.value_pool) == self.state_results(&graph.value_pool)
    }

    /// Write this region's sealed storage into `pool` and stamp the
    /// handles: the interface block
    /// [value params | state params | value results | state results],
    /// contiguous, followed by the nodes block. The single definition
    /// of the sealed layout; construction's seal and compaction's
    /// rebuild both go through it, so a layout change cannot diverge
    /// between them.
    pub(crate) fn write_blocks(
        &mut self,
        pool: &mut ValuePool,
        params: &[ValueId],
        state_params: &[ValueId],
        results: &[ValueId],
        state_results: &[ValueId],
        nodes: &[ValueId],
    ) {
        self.interface_start = pool.extend(params);
        // A pool at exactly u32::MAX entries would hand out a start that
        // collides with the UNSEALED sentinel, leaving the region
        // permanently "open".
        debug_assert!(self.interface_start != Region::UNSEALED);
        pool.extend(state_params);
        pool.extend(results);
        pool.extend(state_results);
        self.nodes_start = pool.extend(nodes);
        self.params_len = u16::try_from(params.len()).expect("region parameter count exceeds u16");
        self.state_params_len =
            u16::try_from(state_params.len()).expect("region state param count exceeds u16");
        self.results_len = u16::try_from(results.len()).expect("region result count exceeds u16");
        self.state_results_len =
            u16::try_from(state_results.len()).expect("region state result count exceeds u16");
        self.nodes_len = u32::try_from(nodes.len()).expect("region node count exceeds u32");
    }
}

/// Growing lists and chain registers of one OPEN region (created but
/// not yet sealed). Region lifetimes nest, but a parent's parameter
/// list keeps growing while children are open (capture-on-demand
/// appends parameters to every region between a binding and its use),
/// so each open region owns its own buffers; the free list recycles
/// them so steady-state construction allocates nothing per region.
/// The lists are plain Vecs: scratches are recycled across regions AND
/// functions (ConstructionScratch), so the free list carries warm heap
/// capacity and steady state allocates nothing -- recycling is what
/// retired the earlier SmallVec choice (its inline slots gain nothing
/// from a free list and cost a 32+ byte memcpy every scratch move; the
/// old "Vec costs ~22k extra allocations" measurement predates
/// cross-function recycling).
#[derive(Debug)]
pub(crate) struct RegionScratch {
    pub(crate) params: Vec<ValueId>,
    pub(crate) nodes: Vec<ValueId>,
    pub(crate) pending_read_states: Vec<State>,
    /// The chains' entry group, recorded at region open; seal writes it
    /// into the state-params tail.
    pub(crate) entry_state: StateGroup,
    /// The chains' running registers: `memory.write` is the newest
    /// write (what a read consumes), `io` the newest io op. Starts as a
    /// copy of the entry group; seal writes it into the state-results
    /// tail.
    pub(crate) exit_state: StateGroup,
}

impl Default for RegionScratch {
    fn default() -> Self {
        RegionScratch {
            params: Vec::new(),
            nodes: Vec::new(),
            pending_read_states: Vec::new(),
            entry_state: StateGroup::INVALID,
            exit_state: StateGroup::INVALID,
        }
    }
}

/// Construction-time scratch for open regions, indexed by RegionId.
/// Freed when construction attaches the finished graph: sealed blocks
/// are write-once, so a pass that changes a region's interface writes
/// replacement blocks at the pool tail and restamps the handles rather
/// than reopening scratch.
#[derive(Debug, Default)]
pub(crate) struct RegionBuilding {
    pub(crate) open: Vec<Option<RegionScratch>>,
    pub(crate) free: Vec<RegionScratch>,
}

impl RegionBuilding {
    pub(crate) fn get_region_scratch_mut(
        &mut self,
        region_id: RegionId,
    ) -> &mut Option<RegionScratch> {
        &mut self.open[region_id.0 as usize]
    }
}
