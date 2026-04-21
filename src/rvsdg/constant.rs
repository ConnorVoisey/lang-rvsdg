use rustc_hash::FxHashMap;

use crate::rvsdg::{GlobalId, types::TypeRef, value::ConstValue};

/// Handle into the constant pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(u32);

/// Span into the ConstIdPool — a contiguous slice of ConstIds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstIdsSpan {
    pub start: u32,
    pub len: u16,
}

/// Flat storage for aggregate element lists, avoiding per-aggregate Vec allocations.
#[derive(Debug, Clone, Default)]
pub struct ConstIdPool(Vec<ConstId>);

impl ConstIdPool {
    pub fn push_slice(&mut self, ids: &[ConstId]) -> ConstIdsSpan {
        let start = self.0.len() as u32;
        self.0.extend_from_slice(ids);
        ConstIdsSpan {
            start,
            len: ids.len() as u16,
        }
    }

    pub fn get(&self, span: ConstIdsSpan) -> &[ConstId] {
        &self.0[span.start as usize..(span.start as usize + span.len as usize)]
    }
}

/// A typed constant value stored in the pool.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstantDef {
    pub ty: TypeRef,
    pub kind: ConstantKind,
}

// TODO: `String(Vec<u8>)` is 24 bytes and forces the entire enum to 24 bytes,
// while most variants fit in 8-16 bytes. A shared byte pool with a BytesSpan
// would bring the enum down to 16 bytes. This pool should be designed alongside
// string interning for struct names, function names, section names, etc. — all
// share the same "intern bytes, get a span back" pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantKind {
    /// Scalar value (integer, float, null, poison)
    Scalar(ConstValue),
    /// Zero-initialized aggregate of the given type
    Zero,
    /// Aggregate constant (struct fields or array elements)
    Aggregate(ConstIdsSpan),
    /// Raw byte string (for string literals like `c"hello\00"`)
    String(Vec<u8>),
    /// Address of a global variable or function
    GlobalAddr(GlobalId),
    /// Undefined value (LLVM's `undef`, distinct from poison)
    Undef,
}

/// Deduplicated storage for all module-level constants.
#[derive(Debug, Default)]
pub struct ConstantPool {
    entries: Vec<ConstantDef>,
    cache: FxHashMap<ConstantDef, ConstId>,
    pub id_pool: ConstIdPool,
}

impl ConstantPool {
    pub fn intern(&mut self, def: ConstantDef) -> ConstId {
        if let Some(&id) = self.cache.get(&def) {
            return id;
        }
        let id = ConstId(self.entries.len() as u32);
        self.entries.push(def.clone());
        self.cache.insert(def, id);
        id
    }

    #[inline]
    pub fn get(&self, id: ConstId) -> &ConstantDef {
        &self.entries[id.0 as usize]
    }

    /// Get the elements of an aggregate constant.
    #[inline]
    pub fn get_aggregate_elements(&self, span: ConstIdsSpan) -> &[ConstId] {
        self.id_pool.get(span)
    }

    /// Convenience: intern a scalar constant.
    pub fn scalar(&mut self, ty: TypeRef, val: ConstValue) -> ConstId {
        self.intern(ConstantDef {
            ty,
            kind: ConstantKind::Scalar(val),
        })
    }

    /// Convenience: intern a zero-initialized constant of the given type.
    pub fn zero(&mut self, ty: TypeRef) -> ConstId {
        self.intern(ConstantDef {
            ty,
            kind: ConstantKind::Zero,
        })
    }

    /// Convenience: intern a byte string constant.
    pub fn string(&mut self, ty: TypeRef, data: Vec<u8>) -> ConstId {
        self.intern(ConstantDef {
            ty,
            kind: ConstantKind::String(data),
        })
    }

    /// Convenience: intern an aggregate constant from existing pool entries.
    pub fn aggregate(&mut self, ty: TypeRef, elements: &[ConstId]) -> ConstId {
        let span = self.id_pool.push_slice(elements);
        self.intern(ConstantDef {
            ty,
            kind: ConstantKind::Aggregate(span),
        })
    }

    /// Convenience: intern a reference to a global's address.
    pub fn global_addr(&mut self, ty: TypeRef, global: GlobalId) -> ConstId {
        self.intern(ConstantDef {
            ty,
            kind: ConstantKind::GlobalAddr(global),
        })
    }
}
