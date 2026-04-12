use crate::rvsdg::{GlobalId, Linkage, RVSDGMod, Visibility, constant::ConstId, types::TypeRef};

// TODO: `name` and `section` are heap-allocated Strings.
// Consider string interning if profiling shows this is a bottleneck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalDef {
    pub name: String,
    pub ty: TypeRef,
    pub initializer: GlobalInit,
    pub is_constant: bool,
    pub linkage: Linkage,
    pub alignment: Option<u32>,
    /// Place this global in a specific object file section (e.g. ".rodata", ".bss")
    pub section: Option<String>,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalInit {
    /// No initializer (external declaration)
    Extern,
    /// Initialized with a constant from the pool
    Init(ConstId),
}

impl RVSDGMod {
    #[inline]
    pub fn define_global(
        &mut self,
        name: String,
        ty: TypeRef,
        initializer: GlobalInit,
        is_constant: bool,
        linkage: Linkage,
    ) -> GlobalId {
        let id = GlobalId(self.globals.len() as u32);
        self.globals.push(GlobalDef {
            name: name.clone(),
            ty,
            initializer,
            is_constant,
            linkage,
            alignment: None,
            section: None,
            visibility: Visibility::default(),
        });
        self.global_map.insert(name, id);
        id
    }

    #[inline]
    pub fn get_global(&self, id: GlobalId) -> &GlobalDef {
        &self.globals[id.0 as usize]
    }

    #[inline]
    pub fn get_global_by_name(&self, name: &str) -> Option<&GlobalDef> {
        self.global_map.get(name).map(|v| self.get_global(*v))
    }
}
