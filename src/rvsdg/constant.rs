use rustc_hash::FxHashMap;

use crate::rvsdg::{GlobalId, types::TypeRef, value::ConstValue};

/// Handle into the constant pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(u32);

/// A typed constant value stored in the pool.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstantDef {
    pub ty: TypeRef,
    pub kind: ConstantKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantKind {
    /// Scalar value (integer, float, null, poison)
    Scalar(ConstValue),
    /// Zero-initialized aggregate of the given type
    Zero,
    /// Aggregate constant (struct fields or array elements)
    Aggregate(Vec<ConstId>),
    /// Raw byte string (for string literals like `c"hello\00"`)
    String(Vec<u8>),
    /// Address of a global variable or function
    GlobalAddr(GlobalId),
    /// Undefined value (LLVM's `undef`, distinct from poison)
    Undef,
}

/// Deduplicated storage for all module-level constants.
#[derive(Default)]
pub struct ConstantPool {
    entries: Vec<ConstantDef>,
    cache: FxHashMap<ConstantDef, ConstId>,
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
    pub fn aggregate(&mut self, ty: TypeRef, elements: Vec<ConstId>) -> ConstId {
        self.intern(ConstantDef {
            ty,
            kind: ConstantKind::Aggregate(elements),
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
