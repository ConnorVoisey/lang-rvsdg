use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrTypeId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrayTypeId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncTypeId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorTypeId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AliasSetId(u32);

pub const BOOL: TypeRef = TypeRef::Scalar(ScalarType::Bool);
pub const I8: TypeRef = TypeRef::Scalar(ScalarType::I8);
pub const I16: TypeRef = TypeRef::Scalar(ScalarType::I16);
pub const I32: TypeRef = TypeRef::Scalar(ScalarType::I32);
pub const I64: TypeRef = TypeRef::Scalar(ScalarType::I64);
pub const I128: TypeRef = TypeRef::Scalar(ScalarType::I128);
pub const F32: TypeRef = TypeRef::Scalar(ScalarType::F32);
pub const F64: TypeRef = TypeRef::Scalar(ScalarType::F64);
pub const VOID: TypeRef = TypeRef::Scalar(ScalarType::Void);

/// Cheap, Copy, no arena lookup needed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,

    I8,
    I16,
    I32,
    I64,
    I128,
    // TODO: add unsigned variants
    F32,
    F64,

    Void,
}
impl ScalarType {
    pub fn is_float(&self) -> bool {
        match self {
            ScalarType::F32 | ScalarType::F64 => true,
            _ => false,
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
            ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::I128 => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeRef {
    State,
    Scalar(ScalarType),
    Ptr(PtrTypeId),
    Array(ArrayTypeId),
    Struct(StructId),
    Vector(VectorTypeId),
    Func(FuncTypeId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PtrType {
    pub pointee: Option<TypeRef>,
    pub alias_set: Option<AliasSetId>,
    pub no_escape: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayType {
    pub element: TypeRef,
    pub len: u64,
}

/// Fixed-width SIMD vector type (e.g. `<4 x i32>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorType {
    /// The scalar element type (must be a scalar or pointer)
    pub element: TypeRef,
    /// Number of lanes
    pub lanes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub params: Vec<TypeRef>,
    pub ret: TypeRef,
    pub is_var_arg: bool,
}

// TODO: `name` fields on StructField and StructDef are heap-allocated Strings.
// Consider string interning if profiling shows this is a bottleneck.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    pub name: Option<String>,
    pub offset: u64,
    pub field_type: TypeRef,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructDef {
    pub name: Option<String>,
    pub fields: Vec<StructField>,
    /// Total size in bytes
    pub size: u64,
}

#[derive(Debug, Default)]
pub struct TypeArena {
    ptrs: Vec<PtrType>,
    arrays: Vec<ArrayType>,
    vectors: Vec<VectorType>,
    funcs: Vec<FuncType>,
    structs: Vec<StructDef>,

    // Deduplication: avoid inserting the same type twice
    ptr_cache: FxHashMap<PtrType, PtrTypeId>,
    array_cache: FxHashMap<ArrayType, ArrayTypeId>,
    vector_cache: FxHashMap<VectorType, VectorTypeId>,
    func_cache: FxHashMap<FuncType, FuncTypeId>,
    struct_cache: FxHashMap<StructDef, StructId>,
}

impl TypeArena {
    pub fn intern_ptr(&mut self, info: PtrType) -> PtrTypeId {
        if let Some(&id) = self.ptr_cache.get(&info) {
            return id;
        }
        let id = PtrTypeId(self.ptrs.len() as u32);
        self.ptrs.push(info.clone());
        self.ptr_cache.insert(info, id);
        id
    }

    pub fn intern_array(&mut self, info: ArrayType) -> ArrayTypeId {
        if let Some(&id) = self.array_cache.get(&info) {
            return id;
        }
        let id = ArrayTypeId(self.arrays.len() as u32);
        self.arrays.push(info.clone());
        self.array_cache.insert(info, id);
        id
    }

    pub fn intern_vector(&mut self, info: VectorType) -> VectorTypeId {
        if let Some(&id) = self.vector_cache.get(&info) {
            return id;
        }
        let id = VectorTypeId(self.vectors.len() as u32);
        self.vectors.push(info);
        self.vector_cache.insert(info, id);
        id
    }

    pub fn intern_fn(&mut self, func_type: FuncType) -> FuncTypeId {
        if let Some(&id) = self.func_cache.get(&func_type) {
            return id;
        }
        let id = FuncTypeId(self.funcs.len() as u32);
        self.funcs.push(func_type.clone());
        self.func_cache.insert(func_type, id);
        id
    }

    pub fn intern_struct(&mut self, struct_def: StructDef) -> StructId {
        if let Some(&id) = self.struct_cache.get(&struct_def) {
            return id;
        }
        let id = StructId(self.structs.len() as u32);
        self.structs.push(struct_def.clone());
        self.struct_cache.insert(struct_def, id);
        id
    }

    pub fn get_ptr(&self, id: PtrTypeId) -> &PtrType {
        &self.ptrs[id.0 as usize]
    }

    pub fn get_array(&self, id: ArrayTypeId) -> &ArrayType {
        &self.arrays[id.0 as usize]
    }

    pub fn get_vector(&self, id: VectorTypeId) -> &VectorType {
        &self.vectors[id.0 as usize]
    }

    pub fn get_fn(&self, id: FuncTypeId) -> &FuncType {
        &self.funcs[id.0 as usize]
    }

    pub fn get_struct(&self, id: StructId) -> &StructDef {
        &self.structs[id.0 as usize]
    }
}
