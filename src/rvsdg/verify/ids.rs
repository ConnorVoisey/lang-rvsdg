use crate::rvsdg::{
    RegionId, ValueId, ValueKind, function_graph::FunctionGraph, module_tables::ModuleTables,
    verify::RVSDGVerificationError,
};

impl FunctionGraph {
    pub(super) fn verify_ids(
        &self,
        module_tables: &ModuleTables,
        errs: &mut Vec<RVSDGVerificationError>,
    ) {
        self.verify_value_ids(module_tables, errs);
    }

    #[inline(always)]
    fn valid_val(&self, errs: &mut Vec<RVSDGVerificationError>, val_id: ValueId) {
        if (val_id.0 as usize) >= self.value_kinds.len() {
            errs.push(RVSDGVerificationError::InvalidValueId(val_id));
        }
    }

    #[inline(always)]
    fn valid_region(&self, errs: &mut Vec<RVSDGVerificationError>, region_id: RegionId) {
        if (region_id.0 as usize) >= self.regions.len() {
            errs.push(RVSDGVerificationError::InvalidRegionId(region_id));
        }
    }

    fn verify_value_ids(
        &self,
        module_tables: &ModuleTables,
        errs: &mut Vec<RVSDGVerificationError>,
    ) {
        for val in self.value_kinds.iter() {
            match *val {
                // Leaf values reference nothing.
                ValueKind::Const(_) | ValueKind::RegionParam { .. } => (),
                ValueKind::ConstPoolRef(const_id) => {
                    if (const_id.0 as usize) >= module_tables.constants.entries.len() {
                        errs.push(RVSDGVerificationError::InvalidConstId(const_id));
                    }
                }
                ValueKind::GlobalRef(global_id) => {
                    if (global_id.0 as usize) >= module_tables.globals.len() {
                        errs.push(RVSDGVerificationError::InvalidGlobalId(global_id));
                    }
                }
                ValueKind::FuncAddr(func_id) => {
                    if (func_id.0 as usize) >= module_tables.functions.len() {
                        errs.push(RVSDGVerificationError::InvalidFnId(func_id));
                    }
                }
                ValueKind::Unary { operand, .. } => {
                    self.valid_val(errs, operand);
                }
                ValueKind::Binary { left, right, .. } => {
                    self.valid_val(errs, left);
                    self.valid_val(errs, right);
                }
                ValueKind::ICmp { left, right, .. } => {
                    self.valid_val(errs, left);
                    self.valid_val(errs, right);
                }
                ValueKind::FCmp { left, right, .. } => {
                    self.valid_val(errs, left);
                    self.valid_val(errs, right);
                }
                ValueKind::Ternary {
                    condition,
                    true_val,
                    false_val,
                } => {
                    self.valid_val(errs, condition);
                    self.valid_val(errs, true_val);
                    self.valid_val(errs, false_val);
                }
                ValueKind::Cast { value, .. } => {
                    self.valid_val(errs, value);
                }
                ValueKind::ExtractLane { vector, index } => {
                    self.valid_val(errs, vector);
                    self.valid_val(errs, index);
                }
                ValueKind::InsertLane {
                    vector,
                    index,
                    value,
                } => {
                    self.valid_val(errs, vector);
                    self.valid_val(errs, index);
                    self.valid_val(errs, value);
                }
                ValueKind::ShuffleLanes { left, right, mask } => {
                    self.valid_val(errs, left);
                    self.valid_val(errs, right);
                    for val_id in self.value_pool.get(mask).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::ExtractField { aggregate, .. } => {
                    self.valid_val(errs, aggregate);
                }
                ValueKind::InsertField {
                    aggregate, value, ..
                } => {
                    self.valid_val(errs, aggregate);
                    self.valid_val(errs, value);
                }
                ValueKind::PtrOffset { base, indices, .. } => {
                    self.valid_val(errs, base);
                    for val_id in self.value_pool.get(indices).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::Load { state, addr, .. } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, addr);
                }
                ValueKind::Store {
                    state, addr, value, ..
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, addr);
                    self.valid_val(errs, value);
                }
                ValueKind::Alloca { state, count, .. } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, count);
                }
                ValueKind::AtomicLoad { state, addr, .. } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, addr);
                }
                ValueKind::AtomicStore {
                    state, addr, value, ..
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, addr);
                    self.valid_val(errs, value);
                }
                ValueKind::AtomicReadModifyWrite {
                    state, addr, value, ..
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, addr);
                    self.valid_val(errs, value);
                }
                ValueKind::CompareAndSwap {
                    state,
                    addr,
                    expected,
                    desired,
                    ..
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, addr);
                    self.valid_val(errs, expected);
                    self.valid_val(errs, desired);
                }
                ValueKind::Fence { state, .. } => {
                    self.valid_val(errs, state.0);
                }
                ValueKind::Freeze { value } => {
                    self.valid_val(errs, value);
                }
                // `arms` hold constant case values and alternative indices, not
                // value ids, so only the matched input is a value reference.
                ValueKind::Match { input, .. } => {
                    self.valid_val(errs, input);
                }
                ValueKind::StateMerge { inputs } => {
                    for val_id in self.value_pool.get(inputs).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::Intrinsic { state, args, .. } => {
                    self.valid_val(errs, state.0);
                    for val_id in self.value_pool.get(args).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::Theta {
                    loop_vars,
                    condition,
                    region_id,
                } => {
                    for val_id in self.value_pool.get(loop_vars).iter() {
                        self.valid_val(errs, *val_id);
                    }
                    self.valid_val(errs, condition);
                    self.valid_region(errs, region_id);
                }
                ValueKind::Gamma {
                    condition,
                    inputs,
                    regions,
                } => {
                    self.valid_val(errs, condition);
                    for val_id in self.value_pool.get(inputs).iter() {
                        self.valid_val(errs, *val_id);
                    }
                    for region_id in self.region_pool.get(regions).iter() {
                        self.valid_region(errs, *region_id);
                    }
                }
                ValueKind::Call {
                    state,
                    io_state,
                    fn_id,
                    sig: _,
                    args,
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, io_state.0);
                    if (fn_id.0 as usize) >= module_tables.functions.len() {
                        errs.push(RVSDGVerificationError::InvalidFnId(fn_id));
                    }
                    for val_id in self.value_pool.get(args).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::CallIndirect {
                    state,
                    io_state,
                    callee,
                    sig: _,
                    args,
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, io_state.0);
                    self.valid_val(errs, callee);
                    for val_id in self.value_pool.get(args).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::Project { call, .. } => {
                    self.valid_val(errs, call);
                }
            }
        }

        // Region interfaces: results and the entry/exit states are ids
        // like any value operand, but they live on Region rather than a
        // ValueKind, so the value loop above never sees them.
        for index in 0..self.regions.len() {
            let region_id = RegionId(index as u32);
            for &result in self.region_results(region_id) {
                self.valid_val(errs, result);
            }
            for &state_param in self.region_state_params(region_id) {
                self.valid_val(errs, state_param);
            }
            for &state_result in self.region_state_results(region_id) {
                self.valid_val(errs, state_result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ConstValue, Linkage, RVSDGMod, ValueId,
        types::{BOOL, I32},
        verify::RVSDGVerificationError,
    };

    /// One function with a gamma, for corrupting region interface ids.
    fn build_gamma_module() -> RVSDGMod {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let picked = rb.gamma(
                predicate,
                &[x, y],
                |rb| Ok(vec![rb.param(0)]),
                |rb| Ok(vec![rb.param(1)]),
            )?;
            Ok(vec![picked.result(0)])
        })
        .unwrap();
        m
    }

    /// Region interface ids get the same range validation as value
    /// operands: a dangling exit-state id is reported as InvalidValueId
    /// (and the verifier stops there, so no downstream pass indexes
    /// out of range with it).
    #[test]
    fn dangling_region_exit_state_id_is_reported() {
        let mut m = build_gamma_module();
        let graph = m.graphs[0].as_mut().unwrap();
        let region = graph.regions[1].clone();
        region.state_results_mut(&mut graph.value_pool)[0] = ValueId(0xFFFF_0000);
        let errs = m.verify();
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::InvalidValueId(_))),
            "expected an invalid-id error, got: {errs:?}"
        );
    }

    /// Same range validation for a dangling entry-state id.
    #[test]
    fn dangling_region_entry_state_id_is_reported() {
        let mut m = build_gamma_module();
        let graph = m.graphs[0].as_mut().unwrap();
        let region = graph.regions[2].clone();
        region.state_params_mut(&mut graph.value_pool)[0] = ValueId(0xFFFF_0000);
        let errs = m.verify();
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::InvalidValueId(_))),
            "expected an invalid-id error, got: {errs:?}"
        );
    }

    /// A region whose seal never happened is a verification error, not
    /// an out-of-bounds panic inside a downstream pass.
    #[test]
    fn unsealed_region_is_reported() {
        let mut m = build_gamma_module();
        let graph = m.graphs[0].as_mut().unwrap();
        graph.regions[1].interface_start = crate::rvsdg::region::Region::UNSEALED;
        let errs = m.verify();
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionUnsealed(_))),
            "expected an unsealed-region error, got: {errs:?}"
        );
    }
}
