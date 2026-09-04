use rustc_hash::FxHashMap;

use crate::rvsdg::{
    ConstId, ConstValue, FuncId, GlobalId, MatchArmPool, RegionId, RegionPool, StateKind, U32Pool,
    Value, ValueId, ValueKind, ValuePool,
    memory_alias::{MemoryFactScratch, origin::MemoryOrigin},
    region::{Region, RegionBuilding},
    state::StateGroup,
    types::TypeRef,
};

/// Construction-worker scratch: one set of heap buffers serving every
/// function a driver builds, in sequence. It RESIDES on the
/// FunctionGraph for the duration of one function (moved in at `new`,
/// back out at `finish_building`) so the graph's internal accessors
/// never change, and it is owned by the construction loop between
/// functions -- the seam that becomes one-scratch-per-worker under
/// parallel construction. Warm capacity from function 1 is what
/// function 1800 builds in.
#[derive(Debug, Default)]
pub struct ConstructionScratch {
    pub(crate) building: RegionBuilding,
    pub(crate) mem_facts: MemoryFactScratch,
}

/// A construct's per-chain state projections; either may be absent once
/// a bypassed chain's projection has been removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstructStateProjections {
    pub memory: Option<ValueId>,
    pub io: Option<ValueId>,
}

#[derive(Debug)]
pub struct FunctionGraph {
    /// The function this graph is the body of. Stamped at construction so
    /// a detached graph identifies itself (error messages, attach-time
    /// cross-checks) before it lands in its module slot; ValueId/RegionId
    /// inside the graph are function-local, this is the module-scoped id.
    pub func_id: FuncId,
    pub value_kinds: Vec<ValueKind>,
    pub value_types: Vec<TypeRef>,
    pub regions: Vec<Region>,
    pub value_pool: ValuePool,
    pub region_pool: RegionPool,
    pub u32_pool: U32Pool,
    pub match_arm_pool: MatchArmPool,
    /// Interned region-free values (constants and symbol references):
    /// one node per distinct value, module-wide, owned by no region.
    /// See [`ValueKind::is_region_free`].
    interned_values: FxHashMap<Value, ValueId>,
    /// Open-region scratch; empty between construction and pass
    /// mutations.
    pub(crate) building: RegionBuilding,
    /// Event scratch for fact collection; same residence pattern as
    /// `building` (see [`ConstructionScratch`]).
    pub(crate) mem_facts: MemoryFactScratch,

    /// Pointer provenance per value indexed by ValueId.
    pub(crate) memory_origins: Vec<MemoryOrigin>,
}

impl FunctionGraph {
    pub fn new(func_id: FuncId, scratch: ConstructionScratch) -> Self {
        Self {
            func_id,
            value_kinds: Vec::default(),
            value_types: Vec::default(),
            regions: Vec::default(),
            value_pool: ValuePool(vec![]),
            region_pool: RegionPool(vec![]),
            u32_pool: U32Pool(vec![]),
            match_arm_pool: MatchArmPool(vec![]),
            interned_values: FxHashMap::default(),
            building: scratch.building,
            mem_facts: scratch.mem_facts,
            memory_origins: Vec::default(),
        }
    }

    // -- Region lifecycle --------------------------------------------
    //
    // A region is CREATED open (its params and nodes grow in the
    // scratch table), then SEALED exactly once at its construct's
    // assembly (define_fn's end for the body region), which writes the
    // interface block [params][results] and the nodes block into the
    // value pool contiguously. The accessors below are the only way to
    // read a region's lists; they serve scratch for open regions and
    // pool segments for sealed ones, so callers never care which phase
    // they are in.

    /// Create a new open region. Its owner is stamped by the construct's
    /// finaliser; its lists accumulate in scratch until
    /// [`FunctionGraph::seal_region`]. `entry` seeds both chains (see
    /// [`FunctionGraph::entry_seeds`] for subregions).
    pub fn create_region(&mut self, entry: StateGroup) -> RegionId {
        let region = RegionId(self.regions.len() as u32);
        self.regions.push(Region::new_open());
        let index = region.0 as usize;
        if self.building.open.len() <= index {
            self.building.open.resize_with(index + 1, || None);
        }
        debug_assert!(self.building.open[index].is_none());
        self.building.open[index] = Some(self.building.free.pop().unwrap_or_default());
        self.set_scratch_entry_state(region, entry);
        region
    }

    /// Write the region's interface block (params from scratch, the
    /// given final value results) and nodes block into the value pool,
    /// and release the scratch. Callable exactly once per open region.
    pub fn seal_region(&mut self, region: RegionId, results: &[ValueId]) {
        // Trailing reads (pending after the last write) fold into the
        // memory exit: a parent-side write after this construct is
        // ordered behind them through the exported merge. Tail order is
        // [memory, io] on both the params and results side.
        let exit_memory = self.state_read_flatten(region);
        let mut scratch = self.building.open[region.0 as usize]
            .take()
            .unwrap_or_else(|| {
                panic!(
                    "{region:?} in {:?} sealed twice or never opened",
                    self.func_id
                )
            });
        let entry = scratch.entry_state;
        let exit_io = scratch.exit_state.io;
        let r = &mut self.regions[region.0 as usize];
        r.write_blocks(
            &mut self.value_pool,
            &scratch.params,
            &[entry.memory.write.0, entry.io.0],
            results,
            &[exit_memory.0, exit_io.0],
            &scratch.nodes,
        );
        scratch.params.clear();
        scratch.nodes.clear();
        self.building.free.push(scratch);
    }

    /// Record an open region's entry state in its scratch, and start the
    /// exit registers as a copy: a region is pure until something
    /// threads a chain, and each chain's current is valid from birth.
    pub(crate) fn set_scratch_entry_state(&mut self, region: RegionId, entry: StateGroup) {
        let scratch = self.get_scratch(region);
        scratch.entry_state = entry;
        scratch.exit_state = entry;
    }

    /// The region's state params: the tail of the params block.
    #[inline]
    pub fn region_state_params(&self, region: RegionId) -> &[ValueId] {
        let r = &self.regions[region.0 as usize];
        assert!(
            r.is_sealed(),
            "state params of unsealed {region:?} in {:?} read",
            self.func_id
        );
        r.state_params(&self.value_pool)
    }

    /// The region's state results: the tail of the results block.
    #[inline]
    pub fn region_state_results(&self, region: RegionId) -> &[ValueId] {
        let r = &self.regions[region.0 as usize];
        assert!(
            r.is_sealed(),
            "state results of unsealed {region:?} in {:?} read",
            self.func_id
        );
        r.state_results(&self.value_pool)
    }

    /// Release the construction scratch, called once when construction
    /// attaches the finished graph. Every region must be sealed by then;
    /// the accessors' sealed fast path never reads the table again.
    /// Salvage the construction scratch from a build that failed
    /// mid-way: open regions are recycled (their buffers cleared into
    /// the free list) and the event scratch is emptied. The caller
    /// discards the graph; the function stays undefined -- graphs[f]
    /// None and empty facts, which the barrier treats exactly like a
    /// declaration (TOP seeding), so a failed body is sound even if
    /// compilation continues past the error.
    pub(crate) fn recover_scratch(mut self) -> ConstructionScratch {
        let mut building = std::mem::take(&mut self.building);
        for slot in building.open.iter_mut() {
            if let Some(mut region_scratch) = slot.take() {
                region_scratch.params.clear();
                region_scratch.nodes.clear();
                region_scratch.pending_read_states.clear();
                building.free.push(region_scratch);
            }
        }
        let mut mem_facts = std::mem::take(&mut self.mem_facts);
        mem_facts.clear();
        ConstructionScratch {
            building,
            mem_facts,
        }
    }

    pub(crate) fn finish_building(&mut self) -> ConstructionScratch {
        debug_assert!(
            self.building.open.iter().all(Option::is_none),
            "graph for {:?} attached with unsealed regions",
            self.func_id
        );
        let mut scratch = ConstructionScratch {
            building: std::mem::take(&mut self.building),
            mem_facts: std::mem::take(&mut self.mem_facts),
        };
        // resolve_facts cleared the events on the normal path; this
        // backstop keeps a recycled scratch clean for callers that
        // attach a graph without resolving.
        scratch.mem_facts.clear();
        scratch
    }

    /// The region's parameters, in input order. The seal check comes
    /// first: it reads the Region entry the segment fields live on
    /// anyway, so the sealed path (every read after construction) never
    /// touches the scratch table. Only open regions -- construction
    /// time -- take the scratch lookup.
    #[inline]
    pub fn region_params(&self, region: RegionId) -> &[ValueId] {
        let r = &self.regions[region.0 as usize];
        if r.is_sealed() {
            return r.params(&self.value_pool);
        }
        self.building.open[region.0 as usize]
            .as_ref()
            .map(|scratch| scratch.params.as_slice())
            .unwrap_or_else(|| panic!("open {region:?} in {:?} has no scratch", self.func_id))
    }

    /// The region's results. Only sealed regions have results (they are
    /// supplied AT seal), so there is no scratch path; an unsealed read
    /// panics with context in release too, like the other mutators.
    #[inline]
    pub fn region_results(&self, region: RegionId) -> &[ValueId] {
        let r = &self.regions[region.0 as usize];
        assert!(
            r.is_sealed(),
            "results of unsealed {region:?} in {:?} read",
            self.func_id
        );
        r.results(&self.value_pool)
    }

    /// The region's nodes, in topological (emission) order. Same sealed
    /// fast path as [`FunctionGraph::region_params`].
    #[inline]
    pub fn region_nodes(&self, region: RegionId) -> &[ValueId] {
        let r = &self.regions[region.0 as usize];
        if r.is_sealed() {
            return self.value_pool.slice(r.nodes_start, r.nodes_len as usize);
        }
        self.building.open[region.0 as usize]
            .as_ref()
            .map(|scratch| scratch.nodes.as_slice())
            .unwrap_or_else(|| panic!("open {region:?} in {:?} has no scratch", self.func_id))
    }

    /// Push a value and record it as a node of `region` (emission
    /// order). The single creation path for region-owned values; the
    /// builder and the state threading both go through it.
    #[inline(always)]
    pub(crate) fn add_region_value(
        &mut self,
        region: RegionId,
        value: Value,
        memory_origin: MemoryOrigin,
    ) -> ValueId {
        let id = ValueId(self.value_kinds.len() as u32);
        self.value_kinds.push(value.kind);
        self.value_types.push(value.ty);
        self.memory_origins.push(memory_origin);
        self.push_region_node(region, id);
        id
    }

    /// Record a value in an open region's node list (emission order).
    #[inline]
    pub(crate) fn push_region_node(&mut self, region: RegionId, id: ValueId) {
        self.building.open[region.0 as usize]
            .as_mut()
            .unwrap_or_else(|| {
                panic!(
                    "node {id:?} added to sealed {region:?} in {:?}",
                    self.func_id
                )
            })
            .nodes
            .push(id);
    }

    /// Append a parameter to an OPEN `region`, returning its value. Used
    /// by the emitter's capture-on-demand: a region acquires an input
    /// the moment its body first reads an outer value, so parameter
    /// values interleave with body values in the value arrays.
    pub(crate) fn append_region_param(
        &mut self,
        region: RegionId,
        ty: TypeRef,
        memory_origin: MemoryOrigin,
    ) -> ValueId {
        let id = ValueId(self.value_kinds.len() as u32);
        let func_id = self.func_id;
        let scratch = self.building.open[region.0 as usize]
            .as_mut()
            .unwrap_or_else(|| panic!("parameter appended to sealed {region:?} in {func_id:?}"));
        let index = scratch.params.len() as u32;
        scratch.params.push(id);
        self.value_kinds
            .push(ValueKind::RegionParam { index, ty, region });
        self.value_types.push(ty);
        self.memory_origins.push(memory_origin);
        id
    }

    /// Replace an open region's parameter list with `params` (construct
    /// assembly aligns every alternative's parameters to one canonical
    /// input order), fixing each parameter value's index and region
    /// fields to its new position.
    pub(crate) fn set_region_params(&mut self, region: RegionId, params: &[ValueId]) {
        for (position, &param) in params.iter().enumerate() {
            let ValueKind::RegionParam {
                index,
                region: param_region,
                ..
            } = &mut self.value_kinds[param.0 as usize]
            else {
                unreachable!("region parameter lists hold only RegionParam values");
            };
            *index = position as u32;
            *param_region = region;
        }
        let func_id = self.func_id;
        let scratch = self.building.open[region.0 as usize]
            .as_mut()
            .unwrap_or_else(|| panic!("params replaced on sealed {region:?} in {func_id:?}"));
        // Every previously appended parameter must reappear in the
        // replacement: a dropped one would linger in the value arrays as
        // a RegionParam with a stale index that no region's list holds.
        debug_assert!(
            scratch.params.iter().all(|old| params.contains(old)),
            "set_region_params on {region:?} in {func_id:?} drops a previously appended param"
        );
        scratch.params.clear();
        scratch.params.extend_from_slice(params);
    }

    #[inline]
    pub fn get_value_kind(&self, value: ValueId) -> &ValueKind {
        &self.value_kinds[value.0 as usize]
    }

    #[inline]
    pub fn get_value_kind_mut(&mut self, value: ValueId) -> &mut ValueKind {
        &mut self.value_kinds[value.0 as usize]
    }

    #[inline]
    pub fn get_value_type(&self, value: ValueId) -> &TypeRef {
        &self.value_types[value.0 as usize]
    }

    #[inline]
    pub fn get_region(&self, region_id: RegionId) -> &Region {
        &self.regions[region_id.0 as usize]
    }

    #[inline]
    pub fn get_region_mut(&mut self, region_id: RegionId) -> &mut Region {
        &mut self.regions[region_id.0 as usize]
    }

    /// Intern a scalar constant: one node per distinct (type, value)
    /// pair, module-wide. Like every region-free value it is pushed into
    /// the value array but into NO region's node list -- region-free
    /// values are usable from every region (the scope verifier exempts
    /// them) and the emitter materialises LLVM constants on demand.
    #[inline]
    pub fn intern_const(&mut self, ty: TypeRef, value: ConstValue) -> ValueId {
        // A pointer-typed constant (null, poison) has no base: Unknown.
        let origin = MemoryOrigin::unknown_if_ptr(ty);
        self.intern_value(
            Value {
                ty,
                kind: ValueKind::Const(value),
            },
            origin,
        )
    }

    /// Intern the address of a global: one node per global, module-wide.
    #[inline]
    pub fn intern_global_ref(&mut self, global: GlobalId, ptr_type: TypeRef) -> ValueId {
        self.intern_value(
            Value {
                ty: ptr_type,
                kind: ValueKind::GlobalRef(global),
            },
            MemoryOrigin::Global(global),
        )
    }

    /// Intern the address of a function: one node per function,
    /// module-wide.
    #[inline]
    pub fn intern_func_addr(&mut self, func: FuncId, ptr_type: TypeRef) -> ValueId {
        // Func is an origin of its own so the address survives
        // derivation: `cond ? &f : &g` must widen into escapes that
        // still NAME both functions.
        self.intern_value(
            Value {
                ty: ptr_type,
                kind: ValueKind::FuncAddr(func),
            },
            MemoryOrigin::Func(func),
        )
    }

    /// Intern a reference to a pooled constant (aggregates, strings,
    /// constant address expressions): one node per pool entry. The
    /// caller resolves the origin (MemoryOrigin::of_constant) -- the
    /// graph cannot see the constant pool.
    #[inline]
    pub fn intern_const_pool_ref(
        &mut self,
        constant: ConstId,
        ty: TypeRef,
        memory_origin: MemoryOrigin,
    ) -> ValueId {
        self.intern_value(
            Value {
                ty,
                kind: ValueKind::ConstPoolRef(constant),
            },
            memory_origin,
        )
    }

    /// Shared by the intern_* constructors above, which are the only
    /// callers and only ever build region-free kinds.
    #[inline]
    fn intern_value(&mut self, value: Value, memory_origin: MemoryOrigin) -> ValueId {
        if let Some(&id) = self.interned_values.get(&value) {
            // An interned value's origin is a function of its kind, so a
            // cache hit must agree with what the caller derived.
            debug_assert_eq!(
                self.memory_origins[id.0 as usize], memory_origin,
                "interned value re-derived with a different origin"
            );
            return id;
        }
        let id = ValueId(self.value_kinds.len() as u32);
        self.value_kinds.push(value.kind);
        self.value_types.push(value.ty);
        self.memory_origins.push(memory_origin);
        self.interned_values.insert(value, id);
        id
    }

    /// Bring the interner along through a compaction pass: drop entries
    /// whose value died and rewrite the survivors' ids, so interning
    /// after the pass never hands out a dangling id. `alive` and
    /// `value_mapper` are indexed by pre-compaction id.
    #[inline]
    pub(crate) fn remap_interned_values(&mut self, alive: &[bool], value_mapper: &[u32]) {
        self.interned_values.retain(|_, id| alive[id.0 as usize]);
        for id in self.interned_values.values_mut() {
            *id = ValueId(value_mapper[id.0 as usize]);
        }
    }

    /// Visit every VALUE operand of `value`, spans expanded through the
    /// pools. State operands are not value operands and are not visited;
    /// this is plain enumeration with no scoping semantics (the scope and
    /// state verifiers have their own, special-cased walks). Exhaustive
    /// over `ValueKind`, so adding a variant forces a decision here.
    #[inline(always)]
    pub fn for_each_value_operand(&self, value: ValueId, mut f: impl FnMut(ValueId)) {
        match self.get_value_kind(value) {
            ValueKind::Unary { operand, .. }
            | ValueKind::Cast { value: operand, .. }
            | ValueKind::Freeze { value: operand }
            | ValueKind::Match { input: operand, .. }
            | ValueKind::ExtractField {
                aggregate: operand, ..
            }
            | ValueKind::Project { call: operand, .. }
            | ValueKind::Alloca { count: operand, .. } => f(*operand),
            ValueKind::Binary { left, right, .. }
            | ValueKind::ICmp { left, right, .. }
            | ValueKind::FCmp { left, right, .. } => {
                f(*left);
                f(*right);
            }
            ValueKind::Ternary {
                condition,
                true_val,
                false_val,
            } => {
                f(*condition);
                f(*true_val);
                f(*false_val);
            }
            ValueKind::ExtractLane { vector, index } => {
                f(*vector);
                f(*index);
            }
            ValueKind::InsertLane {
                vector,
                index,
                value,
            } => {
                f(*vector);
                f(*index);
                f(*value);
            }
            ValueKind::ShuffleLanes { left, right, mask } => {
                f(*left);
                f(*right);
                for &lane in self.value_pool.get(*mask) {
                    f(lane);
                }
            }
            ValueKind::InsertField {
                aggregate, value, ..
            } => {
                f(*aggregate);
                f(*value);
            }
            ValueKind::PtrOffset { base, indices, .. } => {
                f(*base);
                for &index in self.value_pool.get(*indices) {
                    f(index);
                }
            }
            ValueKind::Load { addr, .. } | ValueKind::AtomicLoad { addr, .. } => f(*addr),
            ValueKind::Store { addr, value, .. }
            | ValueKind::AtomicStore { addr, value, .. }
            | ValueKind::AtomicReadModifyWrite { addr, value, .. } => {
                f(*addr);
                f(*value);
            }
            ValueKind::CompareAndSwap {
                addr,
                expected,
                desired,
                ..
            } => {
                f(*addr);
                f(*expected);
                f(*desired);
            }
            ValueKind::Intrinsic { args, .. } | ValueKind::Call { args, .. } => {
                for &arg in self.value_pool.get(*args) {
                    f(arg);
                }
            }
            ValueKind::StateMerge { inputs } => {
                for &input in self.value_pool.get(*inputs) {
                    f(input);
                }
            }
            ValueKind::CallIndirect { callee, args, .. } => {
                f(*callee);
                for &arg in self.value_pool.get(*args) {
                    f(arg);
                }
            }
            ValueKind::Gamma {
                condition, inputs, ..
            } => {
                f(*condition);
                for &input in self.value_pool.get(*inputs) {
                    f(input);
                }
            }
            ValueKind::Theta {
                loop_vars,
                condition,
                ..
            } => {
                f(*condition);
                for &var in self.value_pool.get(*loop_vars) {
                    f(var);
                }
            }
            ValueKind::Const(_)
            | ValueKind::ConstPoolRef(_)
            | ValueKind::GlobalRef(_)
            | ValueKind::FuncAddr(_)
            | ValueKind::Fence { .. }
            | ValueKind::RegionParam { .. } => {}
        }
    }

    /// The subregions of a construct value, empty for non-constructs.
    pub(crate) fn construct_subregions(&self, construct: ValueId) -> &[RegionId] {
        match self.get_value_kind(construct) {
            ValueKind::Gamma { regions, .. } => self.region_pool.get(*regions),
            ValueKind::Theta { region_id, .. } => std::slice::from_ref(region_id),
            _ => &[],
        }
    }

    /// A construct's per-chain state projections, found among its
    /// adjacent projections by their State type. Type-keyed rather than
    /// positional because a bypassed chain's projection is removable:
    /// after dead node elimination a construct may keep one, both, or
    /// neither.
    pub(crate) fn construct_state_projections(
        &self,
        construct: ValueId,
    ) -> ConstructStateProjections {
        let mut found = ConstructStateProjections {
            memory: None,
            io: None,
        };
        let mut id = ValueId(construct.0 + 1);
        while (id.0 as usize) < self.value_kinds.len()
            && matches!(self.get_value_kind(id), ValueKind::Project { call, .. } if *call == construct)
        {
            match self.get_value_type(id) {
                TypeRef::State(StateKind::MemoryRead(_) | StateKind::MemoryWrite(_)) => {
                    found.memory = Some(id);
                }
                TypeRef::State(StateKind::InputOutput) => found.io = Some(id),
                _ => {}
            }
            id = ValueId(id.0 + 1);
        }
        found
    }

    /// The `index`th projection of a multi-output node (loads, calls,
    /// compare-and-swap, gammas, thetas, ...). Every builder allocates a
    /// node's projections immediately after the node itself; this accessor
    /// is the one place that layout is read back, and it CHECKS the
    /// convention instead of trusting it, so a future builder change cannot
    /// silently redirect consumers to the wrong values.
    #[inline]
    pub fn projection_of(&self, node: ValueId, index: u16) -> ValueId {
        let id = ValueId(node.0 + 1 + index as u32);
        match self.get_value_kind(id) {
            ValueKind::Project { call, index: found } if *call == node && *found == index => id,
            other => panic!(
                "projection layout violated: expected Project {{ call: {node:?}, index: {index} }} \
                 at {id:?}, found {other:?}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{ArithFlags, BinaryOp, Linkage, RVSDGMod, RegionId, ValueKind, types::I32};

    /// Seal-layout round trip across the scratch SmallVec inline
    /// boundary: ten parameters and nine body nodes read back through
    /// the accessors identically while the region is open (scratch) and
    /// after the seal (pool blocks). Dead node elimination's compaction
    /// rebuilds exactly this layout, so it must round-trip.
    #[test]
    fn seal_layout_round_trips_across_inline_boundary() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let param_tys = [I32; 10];
        let f = m.declare_fn(String::from("f"), &param_tys, &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let params: Vec<_> = (0..10).map(|i| rb.param(i)).collect();
            let mut acc = params[0];
            for &param in &params[1..] {
                acc = rb.binary(BinaryOp::Add, ArithFlags::default(), acc, param, I32);
            }
            // Mid-construction, the accessors serve the open region from
            // scratch, past the 8-entry inline capacity.
            assert!(!rb.graph.regions[0].is_sealed());
            assert_eq!(rb.graph.region_params(RegionId(0)), params.as_slice());
            let nodes = rb.graph.region_nodes(RegionId(0));
            assert_eq!(nodes.len(), 9);
            assert_eq!(*nodes.last().unwrap(), acc);
            Ok(vec![acc])
        })
        .unwrap();

        // Post-seal, the same lists read back from the pool blocks.
        let graph = m.graphs[0].as_ref().unwrap();
        let region = RegionId(0);
        assert!(graph.regions[0].is_sealed());
        let params = graph.region_params(region);
        assert_eq!(params.len(), 10);
        for (position, &param) in params.iter().enumerate() {
            let ValueKind::RegionParam {
                index, region: r, ..
            } = graph.get_value_kind(param)
            else {
                panic!("params list holds a non-param at {position}");
            };
            assert_eq!(*index, position as u32);
            assert_eq!(*r, region);
        }
        let nodes = graph.region_nodes(region);
        assert_eq!(nodes.len(), 9);
        // Emission order: the accumulator chain was created ascending.
        assert!(nodes.windows(2).all(|pair| pair[0].0 < pair[1].0));
        let results = graph.region_results(region);
        assert_eq!(results, &[*nodes.last().unwrap()]);
        assert!(m.verify().is_empty());
    }
}
