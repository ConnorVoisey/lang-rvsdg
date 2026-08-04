use crate::rvsdg::{
    FuncId, InlineHint, Linkage, RVSDGMod, State, ValueId, Visibility,
    builder::RegionBuilder,
    function_graph::FunctionGraph,
    global::DllStorageClass,
    types::{FuncTypeId, TypeRef},
};
use rustc_hash::FxHashMap;

// TODO: `name` is a heap-allocated String per function.
// Consider string interning if profiling shows this is a bottleneck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FuncId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_types: Vec<TypeRef>,
    pub return_attrs: ParamAttrs,
    /// The ABI signature a call site has when it copies this declaration
    /// verbatim: one attribute set per DECLARED parameter, the declared
    /// return attributes, the declared calling convention. Interned at
    /// declaration time so body construction reads it without touching
    /// the tables. Exact for non-variadic calls; a variadic call site
    /// has more actual arguments than this signature has entries, so
    /// parsers intern the signature from the site instead. None when the
    /// declaration has no LLVM-expressible signature (multi-return).
    pub declared_sig: Option<SignatureId>,

    // Metadata
    pub is_exported: bool,
    pub inline_hint: InlineHint,
    pub linkage_type: Linkage,
    pub calling_convention: CallingConvention,
    pub is_var_arg: bool,
    pub visibility: Visibility,
    pub attrs: FnAttrs,
    /// Windows import/export storage class.
    pub dll_storage_class: DllStorageClass,
}

bitflags::bitflags! {
    /// Function-level boolean attributes. Memory-access behaviour is NOT a
    /// flag -- it is the structured [`MemoryEffects`] on [`FnAttrs`],
    /// mirroring LLVM 16+'s composite memory(...) attribute.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct FnAttrFlags: u32 {
        /// Function never returns (e.g. exit, abort)
        const NO_RETURN            = 1 << 0;
        /// Function never unwinds -- no exceptions or longjmp
        const NO_UNWIND            = 1 << 1;
        /// Function does not recurse, directly or indirectly
        const NO_RECURSE           = 1 << 2;
        /// Function is rarely called -- backend may place in cold section
        const COLD                 = 1 << 3;
        /// Must not be inlined
        const NO_INLINE            = 1 << 4;
        /// Should always be inlined when possible
        const ALWAYS_INLINE        = 1 << 5;
        /// Must use a frame pointer
        const FRAME_POINTER        = 1 << 6;
        /// Stack-smashing protector requested (ssp)
        const STACK_PROTECT        = 1 << 7;
        /// Stack protector required (sspreq)
        const STACK_PROTECT_REQ    = 1 << 8;
        /// Strong stack protector (sspstrong; clang's default hardening)
        const STACK_PROTECT_STRONG = 1 << 9;
        /// Emit an unwind table entry (backtraces through this frame)
        const UWTABLE              = 1 << 10;
        /// Function always returns (never loops forever or aborts)
        const WILL_RETURN          = 1 << 11;
        /// No synchronising operations (atomics, volatile, fences)
        const NO_SYNC              = 1 << 12;
        /// Never frees memory
        const NO_FREE              = 1 << 13;
    }
}

/// How a function may touch each class of memory, mirroring LLVM's
/// composite memory(...) attribute (which replaced the old readnone/
/// readonly/writeonly/argmemonly function attributes in LLVM 16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryEffects {
    /// Everything not covered by the other two classes.
    pub other: ModRef,
    /// Memory reached through pointer arguments.
    pub arg_mem: ModRef,
    /// Memory not reachable from the caller (e.g. errno-like state).
    pub inaccessible_mem: ModRef,
}

/// May-read / may-write for one memory class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModRef {
    NoModRef,
    Ref,
    Mod,
    ModRef,
}

/// Function-level attributes that affect codegen and optimisation.
// TODO: `section` is a heap-allocated String per function that has one.
// Consider string interning if profiling shows this is a bottleneck.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FnAttrs {
    pub flags: FnAttrFlags,
    /// Minimum alignment for the function entry point in bytes.
    pub alignment: Option<u32>,
    /// Object file section (e.g. ".text.cold"). Rarely set.
    pub section: Option<String>,
    /// Structured memory-access behaviour (LLVM's memory(...) attribute);
    /// `None` means unconstrained.
    pub memory: Option<MemoryEffects>,
    /// String attributes carried verbatim (target-cpu, target-features,
    /// frame-pointer, ...). These steer codegen -- dropping target-features
    /// or the stack-protector settings changes the emitted machine code.
    pub string_attrs: Vec<(String, String)>,
}

bitflags::bitflags! {
    /// Parameter/return-value boolean attributes packed into a u16.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct ParamAttrFlags: u16 {
        /// Sign-extend to register width at the call boundary
        const SIGN_EXTEND   = 1 << 0;
        /// Zero-extend to register width at the call boundary
        const ZERO_EXTEND   = 1 << 1;
        /// Pointer does not alias any other pointer visible to callee
        const NO_ALIAS      = 1 << 2;
        /// Pointer is not captured by the callee
        const NO_CAPTURE    = 1 << 3;
        /// Pointer is guaranteed non-null
        const NON_NULL      = 1 << 4;
        /// Pointer is only read through, never written
        const READ_ONLY     = 1 << 5;
        /// Pointer is only written through, never read
        const WRITE_ONLY    = 1 << 6;
        /// Value is neither undef nor poison (clang stamps this on nearly
        /// every parameter)
        const NO_UNDEF      = 1 << 7;
        /// The function returns this argument (e.g. memcpy returns dst)
        const RETURNED      = 1 << 8;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub ty: TypeRef,
    pub flags: ParamAttrFlags,
    /// Rarely-used attributes. None for the common case (zero alloc).
    // TODO: profile real-world programs to determine if interning params
    // into a pool (ParamAttrsId) would be more efficient than boxing here.
    pub extra: Option<Box<ParamAttrsExtra>>,
}

/// Attributes on a parameter or return value that affect ABI and optimisation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ParamAttrs {
    pub flags: ParamAttrFlags,
    /// Rarely-used attributes. None for the common case (zero alloc).
    // TODO: profile real-world programs to determine if interning params
    // into a pool (ParamAttrsId) would be more efficient than boxing here.
    pub extra: Option<Box<ParamAttrsExtra>>,
}

/// Extended parameter attributes that are rarely present.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamAttrsExtra {
    /// Aggregate passed by value -- pointee is copied to the stack
    pub by_value: Option<TypeRef>,
    /// Hidden struct-return pointer -- callee writes return value here
    pub struct_return: Option<TypeRef>,
    /// Pointer argument must be aligned to at least this many bytes
    pub alignment: Option<u32>,
    /// Pointer must point to at least this many dereferenceable bytes
    pub dereferenceable_bytes: Option<u64>,
    /// Range of valid values (lower inclusive, upper exclusive)
    pub range: Option<(i64, i64)>,
}

/// LLVM-compatible calling conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CallingConvention {
    /// Standard C calling convention
    #[default]
    C,
    /// Fast -- allows tail calls, passes args in registers aggressively
    Fast,
    /// Cold -- optimised for rarely-called functions
    Cold,
    /// GHC -- Glasgow Haskell Compiler convention
    GHC,
    /// HiPE -- High Performance Erlang convention
    HiPE,
    /// Preserves most registers across the call
    PreserveMost,
    /// Preserves nearly all registers across the call
    PreserveAll,
    /// Swift calling convention
    Swift,
    /// x86 stdcall (__stdcall)
    X86StdCall,
    /// x86 fastcall (__fastcall)
    X86FastCall,
    /// x86 thiscall (C++ member functions on MSVC)
    X86ThisCall,
    /// x86 vectorcall
    X86VectorCall,
    /// x86 register-based parameter passing
    X86RegCall,
    /// ARM AAPCS (standard ARM convention)
    ArmAAPCS,
    /// ARM AAPCS with VFP registers for float args
    ArmAAPCSVFP,
    /// Win64 (Microsoft x64)
    Win64,
    /// x86-64 System V (Unix x86-64)
    X86_64SysV,
    /// Numbered convention not covered above (LLVM cc N)
    Numbered(u32),
}

/// A dense handle for one interned [`Signature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignatureId(pub(crate) u32);

impl std::fmt::Display for SignatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "sig{}", self.0)
    }
}

/// The complete ABI-bearing signature of an INDIRECT call site: the
/// structural function type plus everything that changes how arguments and
/// results physically move (parameter and return attributes such as
/// `byval`/`sret`/`zeroext`, and the calling convention). Types stay purely
/// structural -- they participate in type equality -- so the ABI lives
/// here instead. Direct calls need none of this (their callee's `Function`
/// is the source of truth), but an indirect call site is the only place its
/// own ABI annotations exist, the same way it is the only place its
/// function type exists under opaque pointers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    pub func_type: FuncTypeId,
    /// One entry per ACTUAL argument at the call site (a variadic call has
    /// more arguments than the function type has parameters).
    pub param_attrs: Vec<ParamAttrs>,
    pub return_attrs: ParamAttrs,
    pub calling_convention: CallingConvention,
}

/// Interned signatures, deduplicated: most indirect call sites in a module
/// share a handful of signatures.
#[derive(Debug, Default)]
pub struct SignatureTable {
    signatures: Vec<Signature>,
    cache: FxHashMap<Signature, SignatureId>,
}

impl SignatureTable {
    pub fn intern(&mut self, signature: Signature) -> SignatureId {
        if let Some(&id) = self.cache.get(&signature) {
            return id;
        }
        let id = SignatureId(self.signatures.len() as u32);
        self.signatures.push(signature.clone());
        self.cache.insert(signature, id);
        id
    }

    pub fn get(&self, id: SignatureId) -> &Signature {
        &self.signatures[id.0 as usize]
    }

    /// Number of distinct interned signatures.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }
}

impl RVSDGMod {
    /// Simple declaration with default attributes and C calling
    /// convention; delegates to the tables and keeps `graphs` FuncId-
    /// indexed by pushing the body slot. See [`ModuleTables::declare_fn`].
    pub fn declare_fn(
        &mut self,
        name: String,
        params: &[TypeRef],
        ret_types: &[TypeRef],
        linkage_type: Linkage,
    ) -> FuncId {
        let id = self
            .tables
            .declare_fn(name, params, ret_types, linkage_type);
        self.graphs.push(None);
        id
    }

    /// Full declaration with explicit control over all function metadata;
    /// delegates to the tables and keeps `graphs` FuncId-indexed.
    pub fn declare_fn_full(&mut self, decl: FnDecl) -> FuncId {
        let id = self.tables.declare_fn_full(decl);
        self.graphs.push(None);
        id
    }

    /// Build `func_id`'s body as a detached [`FunctionGraph`] and attach
    /// it on completion. The graph never coexists half-built with the
    /// module, and construction touches only the tables and its own
    /// graph -- the boundary that later becomes the thread boundary.
    pub fn define_fn(
        &mut self,
        func_id: FuncId,
        rb_fn: impl FnOnce(&mut RegionBuilder, State) -> color_eyre::Result<FnResult>,
    ) -> color_eyre::Result<()> {
        debug_assert_eq!(
            self.graphs.len(),
            self.tables.functions.len(),
            "declare through RVSDGMod so graphs stays FuncId-indexed"
        );
        debug_assert!(self.graphs[func_id.0 as usize].is_none());

        let mut graph = FunctionGraph::new(func_id);
        let mut rb = RegionBuilder::new_from_func(&mut graph, &mut self.tables, func_id);
        let region_id = rb.region_id();
        let state = rb.graph.regions[region_id.0 as usize].entry_state;
        let fn_res = rb_fn(&mut rb, state)?;
        graph.seal_region(region_id, &fn_res.values, fn_res.state);
        graph.finish_building();

        // TODO: if in debug mode check that the return values match the declerations return types
        // Also consider if it is variadic
        self.graphs[func_id.0 as usize] = Some(graph);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_types: Vec<TypeRef>,
    pub return_attrs: ParamAttrs,
    pub linkage_type: Linkage,
    pub calling_convention: CallingConvention,
    pub is_var_arg: bool,
    pub is_exported: bool,
    pub inline_hint: InlineHint,
    pub visibility: Visibility,
    pub attrs: FnAttrs,
    /// Windows import/export storage class.
    pub dll_storage_class: DllStorageClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallResult {
    pub state: State,
    pub first_result: ValueId,
    pub result_count: u16,
}

// TODO: Same short-lived Vec allocation as BranchResult/LoopResult -- see
// builder/mod.rs for the profiling note about SmallVec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnResult {
    pub state: State,
    pub values: Vec<ValueId>,
}

impl CallResult {
    pub fn result(&self, index: u16) -> ValueId {
        debug_assert!(index < self.result_count);
        ValueId(self.first_result.0 + index as u32)
    }
}
