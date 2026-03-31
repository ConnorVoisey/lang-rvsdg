use crate::rvsdg::{
    FuncId, InlineHint, RVSDGMod, State, Value, ValueId, ValueKind, Visibility,
    builder::RegionBuilder,
    types::{ScalarType, TypeRef},
};

// TODO: `name` is a heap-allocated String per function.
// Consider string interning if profiling shows this is a bottleneck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FuncId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_types: Vec<TypeRef>,
    pub return_attrs: ParamAttrs,
    pub lambda_val: Option<ValueId>,

    // Metadata
    pub is_exported: bool,
    pub inline_hint: InlineHint,
    pub linkage_type: FnLinkageType,
    pub calling_convention: CallingConvention,
    pub is_var_arg: bool,
    pub visibility: Visibility,
    pub attrs: FnAttrs,
}

bitflags::bitflags! {
    /// Function-level boolean attributes packed into a u16.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct FnAttrFlags: u16 {
        /// Function never returns (e.g. exit, abort)
        const NO_RETURN                 = 1 << 0;
        /// Function never unwinds — no exceptions or longjmp
        const NO_UNWIND                 = 1 << 1;
        /// Function does not read or write any memory visible to the caller
        const NO_MEMORY                 = 1 << 2;
        /// Function only reads memory, never writes
        const READ_ONLY                 = 1 << 3;
        /// Function only writes memory, never reads
        const WRITE_ONLY                = 1 << 4;
        /// Function does not recurse, directly or indirectly
        const NO_RECURSE                = 1 << 5;
        /// Function does not access memory through pointer arguments
        const NO_ACCESS_ARG_MEMORY      = 1 << 6;
        /// No side effects beyond writing to pointer arguments
        const ONLY_ACCESSES_ARG_MEMORY  = 1 << 7;
        /// Function returns its first argument (e.g. memcpy returns dst)
        const RETURNS_FIRST_ARG         = 1 << 8;
        /// Function is rarely called — backend may place in cold section
        const COLD                      = 1 << 9;
        /// Must not be inlined
        const NO_INLINE                 = 1 << 10;
        /// Should always be inlined when possible
        const ALWAYS_INLINE             = 1 << 11;
        /// Must use a frame pointer
        const FRAME_POINTER             = 1 << 12;
    }
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParamAttrs {
    pub flags: ParamAttrFlags,
    /// Rarely-used attributes. None for the common case (zero alloc).
    // TODO: profile real-world programs to determine if interning params
    // into a pool (ParamAttrsId) would be more efficient than boxing here.
    pub extra: Option<Box<ParamAttrsExtra>>,
}

/// Extended parameter attributes that are rarely present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamAttrsExtra {
    /// Aggregate passed by value — pointee is copied to the stack
    pub by_value: Option<TypeRef>,
    /// Hidden struct-return pointer — callee writes return value here
    pub struct_return: Option<TypeRef>,
    /// Pointer argument must be aligned to at least this many bytes
    pub alignment: Option<u32>,
    /// Pointer must point to at least this many dereferenceable bytes
    pub dereferenceable_bytes: Option<u64>,
    /// Range of valid values (lower inclusive, upper exclusive)
    pub range: Option<(i64, i64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FnLinkageType {
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

/// LLVM-compatible calling conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CallingConvention {
    /// Standard C calling convention
    #[default]
    C,
    /// Fast — allows tail calls, passes args in registers aggressively
    Fast,
    /// Cold — optimised for rarely-called functions
    Cold,
    /// GHC — Glasgow Haskell Compiler convention
    GHC,
    /// HiPE — High Performance Erlang convention
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

impl RVSDGMod {
    /// Simple declaration with default attributes and C calling convention.
    pub fn declare_fn(
        &mut self,
        name: String,
        params: &[TypeRef],
        ret_types: &[TypeRef],
        linkage_type: FnLinkageType,
    ) -> FuncId {
        self.declare_fn_full(FnDecl {
            name,
            params: params
                .iter()
                .map(|&ty| Param {
                    ty,
                    flags: ParamAttrFlags::empty(),
                    extra: None,
                })
                .collect(),
            return_types: ret_types.to_vec(),
            return_attrs: ParamAttrs::default(),
            linkage_type,
            calling_convention: CallingConvention::default(),
            is_var_arg: false,
            is_exported: false,
            inline_hint: InlineHint::Auto,
            visibility: Visibility::default(),
            attrs: FnAttrs::default(),
        })
    }

    /// Full declaration with explicit control over all function metadata.
    pub fn declare_fn_full(&mut self, decl: FnDecl) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        let func = Function {
            id,
            name: decl.name,
            lambda_val: None,
            params: decl.params,
            return_types: decl.return_types,
            return_attrs: decl.return_attrs,
            is_exported: decl.is_exported,
            inline_hint: decl.inline_hint,
            linkage_type: decl.linkage_type,
            calling_convention: decl.calling_convention,
            is_var_arg: decl.is_var_arg,
            visibility: decl.visibility,
            attrs: decl.attrs,
        };
        self.functions.push(func);
        id
    }

    pub fn define_fn(
        &mut self,
        func_id: FuncId,
        rb_fn: impl FnOnce(&mut RegionBuilder, State) -> FnResult,
    ) {
        debug_assert!(self.functions[func_id.0 as usize].lambda_val.is_none());

        let mut rb = RegionBuilder::new_from_func(self, func_id);
        let region_id = rb.region_id();
        let state = rb.graph.regions[region_id.0 as usize].entry_state;
        let fn_res = rb_fn(&mut rb, state);
        let results = rb.graph.value_pool.push_slice(&fn_res.values);
        rb.graph.values.push(Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::RegionResult {
                values: results,
                state: fn_res.state,
            },
        });
        let lambda_val = Value {
            ty: TypeRef::Scalar(ScalarType::Void),
            kind: ValueKind::Lambda {
                region: region_id,
                func_id,
            },
        };
        let lambda_id = rb.add_value(lambda_val);
        self.functions[func_id.0 as usize].lambda_val = Some(lambda_id);
        self.regions[region_id.0 as usize].results = results;

        // TODO: if in debug mode check that the return values match the declerations return types
        // Also consider if it is variadic
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_types: Vec<TypeRef>,
    pub return_attrs: ParamAttrs,
    pub linkage_type: FnLinkageType,
    pub calling_convention: CallingConvention,
    pub is_var_arg: bool,
    pub is_exported: bool,
    pub inline_hint: InlineHint,
    pub visibility: Visibility,
    pub attrs: FnAttrs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallResult {
    pub state: State,
    pub first_result: ValueId,
    pub result_count: u16,
}

// TODO: Same short-lived Vec allocation as BranchResult/LoopResult — see
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
