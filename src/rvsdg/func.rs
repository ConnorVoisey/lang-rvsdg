use crate::rvsdg::{
    FuncId, InlineHint, RVSDGMod, State, Value, ValueId, ValueKind,
    builder::RegionBuilder,
    types::{ScalarType, TypeRef},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FuncId,
    pub name: String,
    pub params: Vec<TypeRef>,
    pub return_types: Vec<TypeRef>,
    pub lambda_val: Option<ValueId>,

    // Metadata
    pub is_exported: bool,
    pub inline_hint: InlineHint,
    pub linkage_type: FnLinkageType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FnLinkageType {
    Internal,
    External,
}

impl RVSDGMod {
    pub fn declare_fn(
        &mut self,
        name: String,
        params: &[TypeRef],
        ret_types: &[TypeRef],
        linkage_type: FnLinkageType,
    ) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        let func = Function {
            id,
            name,
            lambda_val: None,
            // this is very inefficent, but I think we won't be storing params here like this anyway
            params: params.to_vec(),
            return_types: ret_types.to_vec(),
            is_exported: false,
            inline_hint: InlineHint::Auto,
            linkage_type,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallResult {
    pub state: State,
    pub first_result: ValueId,
    pub result_count: u16,
}

#[derive(Debug, Clone)]
pub struct FnResult {
    pub state: State,
    // TODO: this is another allocation that needs to be removed
    pub values: Vec<ValueId>,
}

impl CallResult {
    pub fn result(&self, index: u16) -> ValueId {
        debug_assert!(index < self.result_count);
        ValueId(self.first_result.0 + index as u32)
    }
}
