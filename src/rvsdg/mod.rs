pub mod alias;
pub mod builder;
pub mod constant;
pub mod dump;
pub mod func;
pub mod global;
pub mod lower_to_llvm;
pub mod ops;
pub mod types;
pub mod value;
pub mod verify;

pub use constant::{ConstId, ConstIdPool, ConstIdsSpan, ConstantDef, ConstantKind, ConstantPool};
use func::Function;
pub use global::{GlobalDef, GlobalInit, ThreadLocalMode};
pub use ops::{
    ArithFlags, AtomicRMWOp, BinaryOp, CastOp, FCmpPred, ICmpPred, IntrinsicOp, MemoryOrdering,
    UnaryOp,
};
use rustc_hash::FxHashMap;
pub use target_lexicon::Triple;
use types::{TypeArena, TypeRef};
pub use value::{ConstValue, Value, ValueKind};

#[derive(Debug)]
pub struct RVSDGMod {
    /// Target triple (e.g. x86_64-unknown-linux-gnu)
    pub target: Triple,
    pub mod_name: String,
    /// LLVM data layout string -- encodes pointer sizes, alignments, endianness
    /// for the target. Preserved verbatim for roundtripping through LLVM.
    pub data_layout: String,
    /// Module-level inline assembly (`module asm "..."` lines), preserved
    /// verbatim: it defines real symbols (e.g. hand-written context-switch
    /// routines) that the rest of the module references.
    pub module_asm: String,
    pub types: TypeArena,
    /// Interned ABI signatures for indirect call sites (see
    /// [`func::Signature`]).
    pub signatures: func::SignatureTable,
    pub values: Vec<Value>,
    pub regions: Vec<Region>,
    pub functions: Vec<Function>,
    pub globals: Vec<GlobalDef>,
    pub constants: ConstantPool,
    pub value_pool: ValuePool,
    pub region_pool: RegionPool,
    pub u32_pool: U32Pool,
    pub match_arm_pool: MatchArmPool,
    /// Interned region-free values (constants and symbol references):
    /// one node per distinct value, module-wide, owned by no region.
    /// See [`ValueKind::is_region_free`].
    interned_values: FxHashMap<Value, ValueId>,

    // These maps should probably use &str instead of String
    pub fn_map: FxHashMap<String, FuncId>,
    pub global_map: FxHashMap<String, GlobalId>,
}

impl RVSDGMod {
    pub fn new(mod_name: String, target: Triple, data_layout: String) -> Self {
        Self {
            mod_name,
            target,
            data_layout,
            module_asm: String::new(),
            types: TypeArena::default(),
            signatures: func::SignatureTable::default(),
            values: vec![],
            regions: vec![],
            functions: vec![],
            globals: vec![],
            constants: ConstantPool::default(),
            value_pool: ValuePool(vec![]),
            region_pool: RegionPool(vec![]),
            u32_pool: U32Pool(vec![]),
            match_arm_pool: MatchArmPool(vec![]),
            interned_values: FxHashMap::default(),
            fn_map: FxHashMap::default(),
            global_map: FxHashMap::default(),
        }
    }

    /// Create a module targeting the host platform with an empty data layout.
    pub fn new_host(mod_name: String) -> Self {
        Self::new(mod_name, Triple::host(), String::new())
    }

    #[inline]
    pub fn get(&self, value: ValueId) -> &Value {
        &self.values[value.0 as usize]
    }

    #[inline]
    pub fn get_region(&self, region_id: RegionId) -> &Region {
        &self.regions[region_id.0 as usize]
    }

    #[inline]
    pub fn get_function(&self, func_id: FuncId) -> &Function {
        &self.functions[func_id.0 as usize]
    }

    /// Intern a scalar constant: one node per distinct (type, value)
    /// pair, module-wide. Like every region-free value it is pushed into
    /// the value array but into NO region's node list -- region-free
    /// values are usable from every region (the scope verifier exempts
    /// them) and the emitter materialises LLVM constants on demand, so
    /// region membership would only bloat node lists (measured at 71%
    /// of all values on sqlite3.c before interning).
    pub fn intern_const(&mut self, ty: TypeRef, value: ConstValue) -> ValueId {
        self.intern_value(Value {
            ty,
            kind: ValueKind::Const(value),
        })
    }

    /// Intern the address of a global: one node per global, module-wide.
    pub fn intern_global_ref(&mut self, global: GlobalId, ptr_type: TypeRef) -> ValueId {
        self.intern_value(Value {
            ty: ptr_type,
            kind: ValueKind::GlobalRef(global),
        })
    }

    /// Intern the address of a function: one node per function,
    /// module-wide.
    pub fn intern_func_addr(&mut self, func: FuncId, ptr_type: TypeRef) -> ValueId {
        self.intern_value(Value {
            ty: ptr_type,
            kind: ValueKind::FuncAddr(func),
        })
    }

    /// Intern a reference to a pooled constant (aggregates, strings,
    /// constant address expressions): one node per pool entry.
    pub fn intern_const_pool_ref(&mut self, constant: ConstId, ty: TypeRef) -> ValueId {
        self.intern_value(Value {
            ty,
            kind: ValueKind::ConstPoolRef(constant),
        })
    }

    /// Shared by the intern_* constructors above, which are the only
    /// callers and only ever build region-free kinds.
    fn intern_value(&mut self, value: Value) -> ValueId {
        if let Some(&id) = self.interned_values.get(&value) {
            return id;
        }
        let id = ValueId(self.values.len() as u32);
        self.values.push(value.clone());
        self.interned_values.insert(value, id);
        id
    }

    /// Bring the interner along through a compaction pass: drop entries
    /// whose value died and rewrite the survivors' ids, so interning
    /// after the pass never hands out a dangling id. `alive` and
    /// `value_mapper` are indexed by pre-compaction id.
    pub(crate) fn remap_interned_values(&mut self, alive: &[bool], value_mapper: &[u32]) {
        self.interned_values.retain(|_, id| alive[id.0 as usize]);
        for id in self.interned_values.values_mut() {
            *id = ValueId(value_mapper[id.0 as usize]);
        }
    }

    #[inline]
    pub fn get_region_mut(&mut self, region_id: RegionId) -> &mut Region {
        &mut self.regions[region_id.0 as usize]
    }

    /// Visit every VALUE operand of `value`, spans expanded through the
    /// pools. State operands are not value operands and are not visited;
    /// this is plain enumeration with no scoping semantics (the scope and
    /// state verifiers have their own, special-cased walks). Exhaustive
    /// over `ValueKind`, so adding a variant forces a decision here.
    #[inline(always)]
    pub fn for_each_value_operand(&self, value: ValueId, mut f: impl FnMut(ValueId)) {
        match &self.values[value.0 as usize].kind {
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
            | ValueKind::Lambda { .. }
            | ValueKind::Phi { .. }
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
        match self.values.get(id.0 as usize).map(|value| &value.kind) {
            Some(&ValueKind::Project { call, index: found }) if call == node && found == index => {
                id
            }
            other => panic!(
                "projection layout violated: expected Project {{ call: {node:?}, index: {index} }} \
                 at {id:?}, found {other:?}"
            ),
        }
    }
}

/// Primary handle into the IR. Indexes into RVSDGMod::values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Sentinel for "no value". Deliberately out of range so accidental
    /// use panics at the first indexed access instead of silently
    /// resolving to a real value.
    pub const INVALID: ValueId = ValueId(u32::MAX);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

impl RegionId {
    /// Sentinel for "no region". Same fail-fast rationale as
    /// [`ValueId::INVALID`].
    pub const INVALID: RegionId = RegionId(u32::MAX);
}

#[derive(Debug, Clone, Default)]
pub struct ValuePool(Vec<ValueId>);

impl ValuePool {
    pub fn push_slice(&mut self, values: &[ValueId]) -> ValuesSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(values);
        ValuesSpan {
            start,
            len: values.len() as u16,
        }
    }

    pub fn get(&self, values: ValuesSpan) -> &[ValueId] {
        &self.0[values.start as usize..(values.start as usize + values.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValuesSpan {
    pub start: u32,
    pub len: u16,
}

#[derive(Debug, Clone, Default)]
pub struct RegionPool(Vec<RegionId>);

impl RegionPool {
    pub fn push_slice(&mut self, regions: &[RegionId]) -> RegionsSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(regions);
        RegionsSpan {
            start,
            len: regions.len() as u16,
        }
    }

    pub fn get(&self, span: RegionsSpan) -> &[RegionId] {
        &self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }

    /// Mutable view of a span's contents, for the copy-then-remap-in-
    /// place pattern. Every span is uniquely owned by the field holding
    /// it (`push_slice` always appends), so an owner mutating its span
    /// never aliases another owner's.
    pub fn get_mut(&mut self, span: RegionsSpan) -> &mut [RegionId] {
        &mut self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionsSpan {
    pub start: u32,
    pub len: u16,
}

#[derive(Debug, Clone, Default)]
pub struct U32Pool(Vec<u32>);

impl U32Pool {
    pub fn push_slice(&mut self, values: &[u32]) -> U32Span {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(values);
        U32Span {
            start,
            len: values.len() as u16,
        }
    }

    pub fn get(&self, values: U32Span) -> &[u32] {
        &self.0[values.start as usize..(values.start as usize + values.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct U32Span {
    pub start: u32,
    pub len: u16,
}

/// One arm of a [`ValueKind::Match`]: an integer input value and the control
/// alternative it selects. Stored in [`MatchArmPool`] so `ValueKind` stays a
/// span (all-`Copy`) rather than carrying a heap allocation per node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchArm {
    /// An integer value the matched input may take (e.g. a `switch` case value).
    pub value: i64,
    /// The 0-based control alternative this input value selects.
    pub alternative: u32,
}

#[derive(Debug, Clone, Default)]
pub struct MatchArmPool(Vec<MatchArm>);

impl MatchArmPool {
    pub fn push_slice(&mut self, arms: &[MatchArm]) -> MatchArmSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(arms);
        MatchArmSpan {
            start,
            len: arms.len() as u16,
        }
    }

    pub fn get(&self, span: MatchArmSpan) -> &[MatchArm] {
        &self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }

    /// Total pooled entries (live and dead spans alike).
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchArmSpan {
    pub start: u32,
    pub len: u16,
}

/// State edge -- a newtype over Value for type safety.
/// Prevents accidentally passing a state where data is expected and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct State(pub ValueId);

impl State {
    /// Placeholder for a not-yet-set state slot (regions are created
    /// with this exit state and every finaliser must overwrite it, pure
    /// regions included). Forgetting to set it panics at the first
    /// indexed use instead of silently reading a real value, and the
    /// verifier reports any that survive to a checkpoint.
    pub const INVALID: State = State(ValueId::INVALID);
}

#[derive(Debug, Clone)]
pub struct Region {
    /// The region's parameters, in input order. An explicit list (not a
    /// contiguous span) because construction appends parameters on demand:
    /// the emitter captures outer values into a region while its body is
    /// being built, so parameter values interleave with body values in the
    /// global value array. Consumers identify a parameter by its position
    /// here, never by value-id arithmetic.
    pub params: Vec<ValueId>,
    /// The lambda/gamma/theta/phi value this region belongs to. The
    /// graph only stores the forward direction during emission (the
    /// construct value does not exist until its regions are finished),
    /// so like `exit_state` this is created as [`ValueId::INVALID`] and
    /// stamped by the construct's finaliser; the verifier rejects any
    /// region left unset.
    pub owner: ValueId,
    pub entry_state: State,
    /// Equal to `entry_state` when the region is pure. A field rather
    /// than a results-span entry so slot machinery (projections, phis,
    /// dead slot elimination) never has to special-case a state slot.
    /// Created as [`State::INVALID`]; every finaliser must set it
    /// explicitly, and the verifier rejects any region still unset.
    pub exit_state: State,
    pub results: ValuesSpan,
    /// All values in this region (in topo order)
    pub nodes: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineHint {
    Never,
    Auto,
    Always,
}

/// ELF/Mach-O symbol visibility -- controls linker behavior for shared libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Visibility {
    /// Symbol is visible to other shared objects
    #[default]
    Default,
    /// Symbol is resolved within the defining shared object only
    Hidden,
    /// Like Hidden but the symbol can be overridden by a Default symbol
    Protected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Like Internal but the symbol is also omitted from the symbol table
    Private,
    Internal,
    External,
    /// Merged with other definitions, discarded if unused
    LinkOnce,
    /// Like LinkOnce but preserves the definition for inlining
    LinkOnceODR,
    /// Can be overridden by a stronger definition
    Weak,
    /// Like Weak but preserves the definition for inlining
    WeakODR,
    /// Available for inlining but not emitted if unused
    AvailableExternally,
}
