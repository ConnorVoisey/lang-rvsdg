//! Address resolution: the seed of alias analysis. An address value is
//! walked back through pointer arithmetic (both `PtrOffset` nodes and
//! constant-pool `getelementptr` expressions) to the object it points
//! into, keeping the index path when every step is a compile-time
//! constant. The census consumes this first; optimisation passes that
//! need aliasing judgements (invariant load hoisting, scalar promotion)
//! share the same walk so there is one implementation of "what does
//! this pointer point into".

use crate::rvsdg::{
    FuncId, GlobalId, RVSDGMod, ValueId, ValueKind,
    constant::{ConstId, ConstantKind},
    types::TypeRef,
    value::ConstValue,
};

/// The storage an address points into, as far as a local walk can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseObject {
    /// A named global variable.
    Global(GlobalId),
    /// A stack allocation, identified by its Alloca node.
    Alloca(ValueId),
    /// A function address (not data; loads/stores through it are UB).
    Func(FuncId),
    /// Provenance stops at a region parameter: the pointer came from a
    /// caller or was captured into this region, and the walk does not
    /// cross region boundaries.
    Param(ValueId),
    /// Anything the walk cannot see through: loaded pointers, selects,
    /// call results, integer-to-pointer round trips.
    Unknown(ValueId),
    /// A constant-pool expression that bottoms out in a non-address
    /// constant (e.g. `inttoptr` of a plain integer).
    UnknownConstant(ConstId),
}

/// One `getelementptr` step of a resolved address: the type the indices
/// walk over and the constant index path. Present only when every index
/// in the step is a compile-time integer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstGepStep {
    pub source_type: TypeRef,
    pub indices: Vec<i64>,
}

/// An address resolved to its base plus, when every offset on the way
/// is constant, the full index path (outermost step first). `steps` is
/// `None` as soon as any index is a runtime value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResolvedAddress {
    pub base: BaseObject,
    pub steps: Option<Vec<ConstGepStep>>,
}

impl ResolvedAddress {
    fn unknown(at: ValueId) -> Self {
        ResolvedAddress {
            base: BaseObject::Unknown(at),
            steps: None,
        }
    }

    /// Both addresses resolve to the same base and the same constant
    /// index path: they denote the same memory cell.
    pub fn same_cell(&self, other: &ResolvedAddress) -> bool {
        self.base == other.base && self.steps.is_some() && self.steps == other.steps
    }
}

impl RVSDGMod {
    /// Resolve an address to its base object and constant index path by
    /// walking through `PtrOffset` bases, casts, freezes, and
    /// constant-pool address expressions. Steps are collected outermost
    /// first (the order they would appear in nested source expressions).
    pub fn resolve_address(&self, addr: ValueId) -> ResolvedAddress {
        let mut steps: Vec<ConstGepStep> = Vec::new();
        let mut all_const = true;
        let mut cursor = addr;
        loop {
            match &self.values[cursor.0 as usize].kind {
                ValueKind::GlobalRef(global) => {
                    return self.finish(BaseObject::Global(*global), steps, all_const);
                }
                ValueKind::FuncAddr(func) => {
                    return self.finish(BaseObject::Func(*func), steps, all_const);
                }
                ValueKind::PtrOffset {
                    base,
                    base_type,
                    indices,
                    ..
                } => {
                    let mut path = Vec::new();
                    for &index in self.value_pool.get(*indices) {
                        match self.values[index.0 as usize].kind {
                            ValueKind::Const(ConstValue::Int(k)) => path.push(k),
                            _ => {
                                all_const = false;
                                break;
                            }
                        }
                    }
                    if all_const {
                        steps.push(ConstGepStep {
                            source_type: *base_type,
                            indices: path,
                        });
                    }
                    cursor = *base;
                }
                // Pointer-preserving wrappers. An integer round trip
                // (ptrtoint/inttoptr) keeps the same address, so walking
                // through a cast is sound for base identity.
                ValueKind::Cast { value, .. } | ValueKind::Freeze { value } => cursor = *value,
                ValueKind::Project { call, index } => {
                    if *index == 0
                        && matches!(self.values[call.0 as usize].kind, ValueKind::Alloca { .. })
                    {
                        return self.finish(BaseObject::Alloca(*call), steps, all_const);
                    }
                    return ResolvedAddress::unknown(cursor);
                }
                ValueKind::RegionParam { .. } => {
                    return self.finish(BaseObject::Param(cursor), steps, all_const);
                }
                ValueKind::ConstPoolRef(id) => {
                    return self.resolve_const_address(*id, steps, all_const);
                }
                _ => return ResolvedAddress::unknown(cursor),
            }
        }
    }

    /// Continue an address walk through the constant pool (address-of
    /// expressions in initialisers and interned constant GEPs).
    fn resolve_const_address(
        &self,
        id: ConstId,
        mut steps: Vec<ConstGepStep>,
        mut all_const: bool,
    ) -> ResolvedAddress {
        let mut cursor = id;
        loop {
            let def = self.constants.get(cursor);
            match &def.kind {
                ConstantKind::GlobalAddr(global) => {
                    return self.finish(BaseObject::Global(*global), steps, all_const);
                }
                ConstantKind::FuncAddr(func) => {
                    return self.finish(BaseObject::Func(*func), steps, all_const);
                }
                ConstantKind::GetElementPointer {
                    base,
                    source_type,
                    indices,
                    ..
                } => {
                    let mut path = Vec::new();
                    for &index in self.constants.id_pool.get(*indices) {
                        match self.constants.get(index).kind {
                            ConstantKind::Scalar(ConstValue::Int(k)) => path.push(k),
                            _ => {
                                all_const = false;
                                break;
                            }
                        }
                    }
                    if all_const {
                        steps.push(ConstGepStep {
                            source_type: *source_type,
                            indices: path,
                        });
                    }
                    cursor = *base;
                }
                ConstantKind::Cast { operand, .. } => cursor = *operand,
                ConstantKind::Scalar(_)
                | ConstantKind::Zero
                | ConstantKind::Aggregate(_)
                | ConstantKind::String(_)
                | ConstantKind::Undef => {
                    // Not an address expression (e.g. an inttoptr of a
                    // plain integer bottoms out here).
                    return ResolvedAddress {
                        base: BaseObject::UnknownConstant(cursor),
                        steps: None,
                    };
                }
            }
        }
    }

    /// Steps were collected innermost-last during the walk; store them
    /// outermost-first so equal source expressions resolve identically.
    fn finish(
        &self,
        base: BaseObject,
        mut steps: Vec<ConstGepStep>,
        all_const: bool,
    ) -> ResolvedAddress {
        steps.reverse();
        ResolvedAddress {
            base,
            steps: all_const.then_some(steps),
        }
    }

    /// Conservative may-alias for two ADDRESS values. `false` only when
    /// the accesses are provably disjoint:
    /// - distinct named objects (two different globals, two different
    ///   allocas, or a global vs an alloca), or
    /// - the same base reached through structurally identical constant
    ///   GEP shapes (same step types and index counts) whose index
    ///   paths differ: same-typed walks landing on different elements.
    ///
    /// Same base with runtime offsets, differing GEP shapes, or any
    /// Param/Unknown provenance stays `true`.
    pub fn may_alias(&self, a: ValueId, b: ValueId) -> bool {
        if a == b {
            return true;
        }
        may_alias_resolved(&self.resolve_address(a), &self.resolve_address(b))
    }
}

/// May-alias over already-resolved addresses (callers that scan many
/// accesses resolve once and reuse).
pub fn may_alias_resolved(left: &ResolvedAddress, right: &ResolvedAddress) -> bool {
    use BaseObject::*;
    match (left.base, right.base) {
        (Global(x), Global(y)) if x != y => return false,
        (Alloca(x), Alloca(y)) if x != y => return false,
        (Global(_), Alloca(_)) | (Alloca(_), Global(_)) => return false,
        (Global(_), Global(_)) | (Alloca(_), Alloca(_)) => {}
        _ => return true,
    }
    match (&left.steps, &right.steps) {
        (Some(ls), Some(rs)) => {
            let same_shape = ls.len() == rs.len()
                && ls.iter().zip(rs).all(|(l, r)| {
                    l.source_type == r.source_type && l.indices.len() == r.indices.len()
                });
            if same_shape { ls == rs } else { true }
        }
        _ => true,
    }
}
