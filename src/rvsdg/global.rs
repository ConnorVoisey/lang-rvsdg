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
    /// Minimum alignment in bytes. `None` leaves the target's ABI
    /// alignment; over-aligned globals (alignas, cache-line placement,
    /// wide atomics) depend on this surviving to the emitted module.
    pub alignment: Option<u32>,
    /// Place this global in a specific object file section (e.g. ".rodata", ".bss")
    pub section: Option<String>,
    pub visibility: Visibility,
    /// Thread-local storage mode. Anything but `NotThreadLocal` gives each
    /// thread its own copy of the global; the mode picks the TLS access
    /// model. Must be preserved through lowering: accesses go through the
    /// `llvm.threadlocal.address` intrinsic, whose operand LLVM requires
    /// to actually be thread-local.
    pub thread_local: ThreadLocalMode,
    /// Target address space (0 is the default; nonzero matters for GPU and
    /// segmented-memory targets).
    pub addr_space: u32,
    /// Whether the global's address is significant (used for constant
    /// merging; `unnamed_addr` in LLVM).
    pub unnamed_addr: UnnamedAddr,
    /// Windows import/export storage class.
    pub dll_storage_class: DllStorageClass,
}

/// Whether a global's ADDRESS is observable, mirroring LLVM's
/// unnamed_addr: an address that is never compared can be merged with any
/// other identical constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum UnnamedAddr {
    /// The address is significant (the default).
    #[default]
    None,
    /// The address is insignificant within the module (`local_unnamed_addr`).
    Local,
    /// The address is globally insignificant (`unnamed_addr`).
    Global,
}

/// Windows DLL import/export storage class, mirroring LLVM's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DllStorageClass {
    #[default]
    Default,
    Import,
    Export,
}

/// The thread-local storage access model of a global, mirroring LLVM's
/// (see the LLVM LangRef on thread-local storage). `NotThreadLocal` is the
/// ordinary shared global.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThreadLocalMode {
    #[default]
    NotThreadLocal,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalInit {
    /// No initializer (external declaration)
    Extern,
    /// Initialized with a constant from the pool
    Init(ConstId),
}

impl GlobalDef {
    /// A plain module-internal global with default attributes: the
    /// convenience for hand-built graphs (tests, examples). The frontend
    /// never uses this -- it constructs the full literal from an
    /// exhaustive destructure of the source global, so no attribute can
    /// default invisibly on the parse path.
    pub fn plain(
        name: String,
        ty: TypeRef,
        initializer: GlobalInit,
        is_constant: bool,
        linkage: Linkage,
    ) -> Self {
        Self {
            name,
            ty,
            initializer,
            is_constant,
            linkage,
            alignment: None,
            section: None,
            visibility: Visibility::default(),
            thread_local: ThreadLocalMode::default(),
            addr_space: 0,
            unnamed_addr: UnnamedAddr::default(),
            dll_storage_class: DllStorageClass::default(),
        }
    }
}

impl RVSDGMod {
    /// Register a global. Takes the complete definition -- there are no
    /// hidden defaults here, which is what kept alignment/section/
    /// visibility silently unpopulated for months under the old
    /// five-argument signature.
    #[inline]
    pub fn define_global(&mut self, def: GlobalDef) -> GlobalId {
        let id = GlobalId(self.globals.len() as u32);
        self.global_map.insert(def.name.clone(), id);
        self.globals.push(def);
        id
    }

    /// Convenience over [`GlobalDef::plain`] for hand-built graphs.
    #[inline]
    pub fn define_global_plain(
        &mut self,
        name: String,
        ty: TypeRef,
        initializer: GlobalInit,
        is_constant: bool,
        linkage: Linkage,
    ) -> GlobalId {
        self.define_global(GlobalDef::plain(
            name,
            ty,
            initializer,
            is_constant,
            linkage,
        ))
    }

    /// Set (or replace) a global's initializer after it was declared.
    /// Used by the two-pass global lowering: every global is registered
    /// first so initializers can forward-reference globals declared later.
    #[inline]
    pub fn set_global_init(&mut self, id: GlobalId, initializer: GlobalInit) {
        self.globals[id.0 as usize].initializer = initializer;
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
