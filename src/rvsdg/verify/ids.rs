use crate::rvsdg::{RVSDGMod, RegionId, ValueId, ValueKind, verify::RVSDGVerificationError};

impl RVSDGMod {
    pub(super) fn verify_ids(&self, errs: &mut Vec<RVSDGVerificationError>) {
        self.verify_value_ids(errs);
    }

    #[inline(always)]
    fn valid_val(&self, errs: &mut Vec<RVSDGVerificationError>, val_id: ValueId) {
        if (val_id.0 as usize) >= self.values.len() {
            errs.push(RVSDGVerificationError::InvalidValueId(val_id));
        }
    }

    #[inline(always)]
    fn valid_region(&self, errs: &mut Vec<RVSDGVerificationError>, region_id: RegionId) {
        if (region_id.0 as usize) >= self.regions.len() {
            errs.push(RVSDGVerificationError::InvalidRegionId(region_id));
        }
    }

    fn verify_value_ids(&self, errs: &mut Vec<RVSDGVerificationError>) {
        for val in self.values.iter() {
            match val.kind {
                // Leaf values reference nothing.
                ValueKind::Const(_) | ValueKind::RegionParam { .. } => (),
                ValueKind::ConstPoolRef(const_id) => {
                    if (const_id.0 as usize) >= self.constants.entries.len() {
                        errs.push(RVSDGVerificationError::InvalidConstId(const_id));
                    }
                }
                ValueKind::GlobalRef(global_id) => {
                    if (global_id.0 as usize) >= self.globals.len() {
                        errs.push(RVSDGVerificationError::InvalidGlobalId(global_id));
                    }
                }
                ValueKind::FuncAddr(func_id) => {
                    if (func_id.0 as usize) >= self.functions.len() {
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
                ValueKind::Intrinsic { state, args, .. } => {
                    self.valid_val(errs, state.0);
                    for val_id in self.value_pool.get(args).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::Lambda { region, func_id } => {
                    self.valid_region(errs, region);
                    if (func_id.0 as usize) >= self.functions.len() {
                        errs.push(RVSDGVerificationError::InvalidFnId(func_id));
                    }
                }
                ValueKind::Theta {
                    loop_vars,
                    condition,
                    state,
                    region_id,
                } => {
                    for val_id in self.value_pool.get(loop_vars).iter() {
                        self.valid_val(errs, *val_id);
                    }
                    self.valid_val(errs, condition);
                    self.valid_val(errs, state.0);
                    self.valid_region(errs, region_id);
                }
                ValueKind::Gamma {
                    condition,
                    inputs,
                    state,
                    regions,
                } => {
                    self.valid_val(errs, condition);
                    for val_id in self.value_pool.get(inputs).iter() {
                        self.valid_val(errs, *val_id);
                    }
                    self.valid_val(errs, state.0);
                    for region_id in self.region_pool.get(regions).iter() {
                        self.valid_region(errs, *region_id);
                    }
                }
                ValueKind::Phi { region, .. } => {
                    self.valid_region(errs, region);
                }
                ValueKind::Call { state, fn_id, args } => {
                    self.valid_val(errs, state.0);
                    if (fn_id.0 as usize) >= self.functions.len() {
                        errs.push(RVSDGVerificationError::InvalidFnId(fn_id));
                    }
                    for val_id in self.value_pool.get(args).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::CallIndirect {
                    state,
                    callee,
                    args,
                } => {
                    self.valid_val(errs, state.0);
                    self.valid_val(errs, callee);
                    for val_id in self.value_pool.get(args).iter() {
                        self.valid_val(errs, *val_id);
                    }
                }
                ValueKind::Project { call, .. } => {
                    self.valid_val(errs, call);
                }
                ValueKind::RegionResult { values, state } => {
                    for val_id in self.value_pool.get(values).iter() {
                        self.valid_val(errs, *val_id);
                    }
                    self.valid_val(errs, state.0);
                }
            }
        }
    }
}
