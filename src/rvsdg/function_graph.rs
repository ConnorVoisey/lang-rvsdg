use rustc_hash::FxHashMap;

use crate::rvsdg::{
    ConstId, ConstValue, FuncId, GlobalId, MatchArmPool, Region, RegionId, RegionPool, U32Pool,
    Value, ValueId, ValueKind, ValuePool, types::TypeRef,
};

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
}

impl FunctionGraph {
    pub fn new(func_id: FuncId) -> Self {
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
        }
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
        self.intern_value(Value {
            ty,
            kind: ValueKind::Const(value),
        })
    }

    /// Intern the address of a global: one node per global, module-wide.
    #[inline]
    pub fn intern_global_ref(&mut self, global: GlobalId, ptr_type: TypeRef) -> ValueId {
        self.intern_value(Value {
            ty: ptr_type,
            kind: ValueKind::GlobalRef(global),
        })
    }

    /// Intern the address of a function: one node per function,
    /// module-wide.
    #[inline]
    pub fn intern_func_addr(&mut self, func: FuncId, ptr_type: TypeRef) -> ValueId {
        self.intern_value(Value {
            ty: ptr_type,
            kind: ValueKind::FuncAddr(func),
        })
    }

    /// Intern a reference to a pooled constant (aggregates, strings,
    /// constant address expressions): one node per pool entry.
    #[inline]
    pub fn intern_const_pool_ref(&mut self, constant: ConstId, ty: TypeRef) -> ValueId {
        self.intern_value(Value {
            ty,
            kind: ValueKind::ConstPoolRef(constant),
        })
    }

    /// Shared by the intern_* constructors above, which are the only
    /// callers and only ever build region-free kinds.
    #[inline]
    fn intern_value(&mut self, value: Value) -> ValueId {
        if let Some(&id) = self.interned_values.get(&value) {
            return id;
        }
        let id = ValueId(self.value_kinds.len() as u32);
        self.value_kinds.push(value.kind);
        self.value_types.push(value.ty);
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
            ValueKind::RegionResult { values, .. } => {
                for &result in self.value_pool.get(*values) {
                    f(result);
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
