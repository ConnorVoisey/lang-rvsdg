use crate::rvsdg::{constant::ConstId, types::TypeRef};

pub struct GlobalDef {
    pub name: String,
    pub ty: TypeRef,
    pub initializer: GlobalInit,
    pub is_constant: bool,
    pub linkage: GlobalLinkage,
    pub alignment: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalInit {
    /// No initializer (external declaration)
    Extern,
    /// Initialized with a constant from the pool
    Init(ConstId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalLinkage {
    /// Visible only within this module
    Internal,
    /// Visible to the linker
    External,
    /// Merge with other definitions of the same name
    LinkOnce,
    /// Like LinkOnce but can be discarded if unused
    Weak,
    /// Common symbol (tentative definition in C)
    Common,
}
