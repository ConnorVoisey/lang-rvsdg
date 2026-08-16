use crate::rvsdg::{
    RegionId, Value, ValueId, ValueKind, function_graph::FunctionGraph, region::RegionScratch,
    types::TypeRef,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AliasClassId(pub(crate) u32);

/// State edge -- a newtype over Value for type safety.
/// Prevents accidentally passing a state where data is expected and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State(pub(crate) ValueId);

impl State {
    /// Placeholder for a not-yet-known state value: `new_from_func`
    /// opens the function region with INVALID seeds and stamps the real
    /// entry group once the state parameters exist. Any INVALID that
    /// leaks into an indexed use panics at the first access instead of
    /// silently reading a real value.
    pub const INVALID: State = State(ValueId::INVALID);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateKind {
    MemoryRead(AliasClassId),
    MemoryWrite(AliasClassId),
    InputOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStates {
    /// Write-only until the alias-class split, which gives each memory
    /// chain a read face (later writes wait on it) beside the write
    /// face (later reads wait only on that).
    pub read: State,
    pub write: State,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateGroup {
    pub memory: MemoryStates,
    pub io: State,
}

impl StateGroup {
    pub(crate) const INVALID: StateGroup = StateGroup {
        memory: MemoryStates {
            read: State::INVALID,
            write: State::INVALID,
        },
        io: State::INVALID,
    };
}

impl FunctionGraph {
    /// The state a reading op must consume: the chain's newest write
    /// (the entry state until the first write). Reads never advance it.
    #[inline(always)]
    pub(crate) fn state_current(&mut self, region_id: RegionId) -> State {
        self.get_scratch(region_id).exit_state.memory.write
    }

    /// Record a read: the op joins the pending list; current is
    /// unchanged (reads fan out from the last write, they don't chain).
    #[inline(always)]
    pub(crate) fn state_read(&mut self, region_id: RegionId, value_id: ValueId) -> State {
        let state = State(value_id);
        self.get_scratch(region_id).pending_read_states.push(state);
        state
    }

    pub(crate) fn get_scratch(&mut self, region_id: RegionId) -> &mut RegionScratch {
        let func_id = self.func_id;
        self.building
            .get_region_scratch_mut(region_id)
            .as_mut()
            .unwrap_or_else(|| panic!("{region_id:?} in {func_id:?} has no open scratch"))
    }

    /// The state a writing op must consume, with every pending read
    /// ordered before it: none pending -> current; one -> that read,
    /// drained; several -> a new StateMerge over them, drained.
    #[inline(always)]
    pub(crate) fn state_read_flatten(&mut self, region_id: RegionId) -> State {
        // One scratch lookup serves every arm; the borrow is taken
        // through `building` alone so the merge arm can write the
        // disjoint value pool while it lives.
        let func_id = self.func_id;
        let scratch = self
            .building
            .get_region_scratch_mut(region_id)
            .as_mut()
            .unwrap_or_else(|| panic!("{region_id:?} in {func_id:?} has no open scratch"));
        match scratch.pending_read_states.len() {
            0 => scratch.exit_state.memory.write,
            1 => {
                let state = scratch.pending_read_states[0];
                scratch.pending_read_states.clear();
                state
            }
            _ => {
                let inputs = self
                    .value_pool
                    .push_iter(scratch.pending_read_states.drain(..).map(|state| state.0));
                let value = self.add_region_value(
                    region_id,
                    Value {
                        ty: TypeRef::State(StateKind::MemoryRead(AliasClassId(0))),
                        kind: ValueKind::StateMerge { inputs },
                    },
                );
                State(value)
            }
        }
    }

    /// Thread a writing op: `make` receives the state the op must
    /// consume (pending reads flattened) and returns the op's Value,
    /// which is added to the region and becomes the chain's new
    /// current.
    #[inline(always)]
    pub(crate) fn state_write(
        &mut self,
        region_id: RegionId,
        make: impl FnOnce(State) -> Value,
    ) -> State {
        let input_state = self.state_read_flatten(region_id);
        let out_state = State(self.add_region_value(region_id, make(input_state)));
        let memory = &mut self.get_scratch(region_id).exit_state.memory;
        memory.write = out_state;
        memory.read = out_state;
        out_state
    }

    /// The io chain's current value (the entry state until the first
    /// io op). Io ops consume it directly; there is no read fan-out on
    /// the io chain -- every io op is a write.
    #[inline(always)]
    pub(crate) fn state_io_current(&mut self, region_id: RegionId) -> State {
        self.get_scratch(region_id).exit_state.io
    }

    /// Advance the io chain: `value_id` (an io-performing op) becomes
    /// its new current.
    #[inline(always)]
    pub(crate) fn state_io(&mut self, region_id: RegionId, value_id: ValueId) -> State {
        let state = State(value_id);
        self.get_scratch(region_id).exit_state.io = state;
        state
    }

    /// Advance both chains onto a construct's state projections at
    /// assembly; the parent's chains continue on them.
    pub(crate) fn state_construct_outputs(
        &mut self,
        region_id: RegionId,
        memory: State,
        io: State,
    ) {
        let group = &mut self.get_scratch(region_id).exit_state;
        group.memory.write = memory;
        group.memory.read = memory;
        group.io = io;
    }

    /// Check the seeding/assembly contract at a construct's assembly:
    /// nothing may have run in the parent since its subregions were
    /// seeded. A violation would leave the parent-side op's output state
    /// unconsumed once the registers advance onto the construct's
    /// projections -- the chain would silently skip a real effect.
    /// `subregion` is any of the construct's (sealed) subregions; its
    /// entry tail records the seeds.
    pub(crate) fn debug_assert_seeds_are_current(&mut self, parent: RegionId, subregion: RegionId) {
        if cfg!(debug_assertions) {
            let seeds = self.region_state_params(subregion);
            let (seed_memory, seed_io) = (seeds[0], seeds[1]);
            let scratch = self.get_scratch(parent);
            debug_assert!(
                scratch.exit_state.memory.write.0 == seed_memory
                    && scratch.exit_state.io.0 == seed_io,
                "construct assembled after parent-side state ops: the parent's chains moved \
                 past the subregion seeds, so those ops' effects would be silently skipped"
            );
            debug_assert!(
                scratch.pending_read_states.is_empty(),
                "construct assembled with parent reads pending; subregions must be seeded \
                 after them"
            );
        }
    }

    /// The seeds a subregion opens with: the parent's pending reads
    /// flattened, plus the io current. The entry tails these seeds
    /// become are THE record of what the enclosing construct chains
    /// from, so the flatten happens here (ordering the construct behind
    /// every pending read) and its result advances the parent's current
    /// -- sibling arms seeded later must receive the identical seed, and
    /// nothing else runs in the parent until the construct's own state
    /// projections take over at assembly.
    pub(crate) fn entry_seeds(&mut self, parent: RegionId) -> StateGroup {
        let memory = self.state_read_flatten(parent);
        let group = &mut self.get_scratch(parent).exit_state;
        group.memory.write = memory;
        group.memory.read = memory;
        *group
    }
}
