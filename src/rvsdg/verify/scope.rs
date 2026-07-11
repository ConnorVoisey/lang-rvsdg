//! **Region scoping** -- the "values flow through graph edges" rule. A
//! node may only use values that are visible in its region: an earlier
//! node of the SAME region, one of the region's parameters, or a
//! region-free value (constants and symbol references, which denote the
//! same thing everywhere). Anything else -- reaching into a sibling gamma
//! arm, reading an enclosing value without a capture parameter -- is the
//! cross-region reuse bug class, which lowers to LLVM values that dominate
//! nothing and miscompiles or crashes far from the cause.
//!
//! Two DESIGNED exceptions:
//! - STATE operands are exempt from the region rule HERE and verified by
//!   the sibling state pass (`verify/state.rs`): state is the compile-time
//!   sequencing chain, and subregions receive the enclosing state value
//!   through `Region::entry_state` rather than a parameter, so state edges
//!   cross region boundaries by construction and need their own rule.
//! - A theta's repetition `condition` lives INSIDE its body region (it is
//!   the body's predicate slot), so it is checked against the body, not
//!   against the theta node's own region.

use crate::rvsdg::{RVSDGMod, RegionId, ValueId, ValueKind, verify::RVSDGVerificationError};

/// Where a value lives, built once over the whole graph and shared by the
/// scope and state passes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Owner {
    /// Not in any region's nodes or params: region results, entry states,
    /// and other structural values that are never ordinary operands.
    Unowned,
    /// `region.nodes[position]`.
    Node { region: u32, position: u32 },
    /// One of `region.params`.
    Param { region: u32 },
}

impl RVSDGMod {
    /// Build the value-to-region ownership map the scope and state passes
    /// share. A value appearing in more than one region's node/param lists
    /// is reported here; ownership then stays with the FIRST region seen,
    /// so any follow-up out-of-scope errors naming the other region are
    /// cascade noise from this one.
    pub(super) fn build_value_ownership(
        &self,
        errs: &mut Vec<RVSDGVerificationError>,
    ) -> Vec<Owner> {
        let mut owner = vec![Owner::Unowned; self.values.len()];
        for (region_index, region) in self.regions.iter().enumerate() {
            for (position, &value) in region.nodes.iter().enumerate() {
                if owner[value.0 as usize] != Owner::Unowned {
                    errs.push(RVSDGVerificationError::ValueInMultipleRegions(value));
                    continue;
                }
                owner[value.0 as usize] = Owner::Node {
                    region: region_index as u32,
                    position: position as u32,
                };
            }
            for &param in &region.params {
                if owner[param.0 as usize] != Owner::Unowned {
                    errs.push(RVSDGVerificationError::ValueInMultipleRegions(param));
                    continue;
                }
                owner[param.0 as usize] = Owner::Param {
                    region: region_index as u32,
                };
            }
        }
        owner
    }

    pub(super) fn verify_scope(&self, owner: &[Owner], errs: &mut Vec<RVSDGVerificationError>) {
        // A value visible everywhere: constants and symbol references
        // denote the same thing in any region, so they are exempt from
        // ownership (the emitter materialises them where needed).
        let region_free = |value: ValueId| {
            matches!(
                self.values[value.0 as usize].kind,
                ValueKind::Const(_)
                    | ValueKind::ConstPoolRef(_)
                    | ValueKind::GlobalRef(_)
                    | ValueKind::FuncAddr(_)
            )
        };

        let check = |errs: &mut Vec<RVSDGVerificationError>,
                     user: ValueId,
                     user_region: u32,
                     user_position: u32,
                     operand: ValueId| {
            if region_free(operand) {
                return;
            }
            match owner[operand.0 as usize] {
                Owner::Node { region, position } if region == user_region => {
                    if position >= user_position {
                        errs.push(RVSDGVerificationError::ValueUsedBeforeDefinition {
                            user,
                            operand,
                        });
                    }
                }
                Owner::Param { region } if region == user_region => {}
                _ => errs.push(RVSDGVerificationError::ValueUsedOutOfScope {
                    user,
                    operand,
                    region: RegionId(user_region),
                }),
            }
        };

        for (region_index, region) in self.regions.iter().enumerate() {
            let user_region = region_index as u32;
            for (position, &user) in region.nodes.iter().enumerate() {
                let user_position = position as u32;
                let visit = |errs: &mut Vec<RVSDGVerificationError>, operand: ValueId| {
                    check(errs, user, user_region, user_position, operand);
                };
                // One exhaustive walk over value operands. STATE fields are
                // deliberately never visited here (the state pass covers
                // them); spans are expanded through the pools like the
                // other verifiers.
                match &self.values[user.0 as usize].kind {
                    ValueKind::Unary { operand, .. }
                    | ValueKind::Cast { value: operand, .. }
                    | ValueKind::Freeze { value: operand }
                    | ValueKind::Match { input: operand, .. } => visit(errs, *operand),
                    ValueKind::Binary { left, right, .. }
                    | ValueKind::ICmp { left, right, .. }
                    | ValueKind::FCmp { left, right, .. } => {
                        visit(errs, *left);
                        visit(errs, *right);
                    }
                    ValueKind::Ternary {
                        condition,
                        true_val,
                        false_val,
                    } => {
                        visit(errs, *condition);
                        visit(errs, *true_val);
                        visit(errs, *false_val);
                    }
                    ValueKind::ExtractLane { vector, index } => {
                        visit(errs, *vector);
                        visit(errs, *index);
                    }
                    ValueKind::InsertLane {
                        vector,
                        index,
                        value,
                    } => {
                        visit(errs, *vector);
                        visit(errs, *index);
                        visit(errs, *value);
                    }
                    ValueKind::ShuffleLanes { left, right, mask } => {
                        visit(errs, *left);
                        visit(errs, *right);
                        for &lane in self.value_pool.get(*mask) {
                            visit(errs, lane);
                        }
                    }
                    ValueKind::ExtractField { aggregate, .. } => visit(errs, *aggregate),
                    ValueKind::InsertField {
                        aggregate, value, ..
                    } => {
                        visit(errs, *aggregate);
                        visit(errs, *value);
                    }
                    ValueKind::PtrOffset { base, indices, .. } => {
                        visit(errs, *base);
                        for &index in self.value_pool.get(*indices) {
                            visit(errs, index);
                        }
                    }
                    ValueKind::Load { addr, .. } | ValueKind::AtomicLoad { addr, .. } => {
                        visit(errs, *addr);
                    }
                    ValueKind::Store { addr, value, .. }
                    | ValueKind::AtomicStore { addr, value, .. }
                    | ValueKind::AtomicReadModifyWrite { addr, value, .. } => {
                        visit(errs, *addr);
                        visit(errs, *value);
                    }
                    ValueKind::Alloca { count, .. } => visit(errs, *count),
                    ValueKind::CompareAndSwap {
                        addr,
                        expected,
                        desired,
                        ..
                    } => {
                        visit(errs, *addr);
                        visit(errs, *expected);
                        visit(errs, *desired);
                    }
                    ValueKind::Intrinsic { args, .. } | ValueKind::Call { args, .. } => {
                        for &arg in self.value_pool.get(*args) {
                            visit(errs, arg);
                        }
                    }
                    ValueKind::CallIndirect { callee, args, .. } => {
                        visit(errs, *callee);
                        for &arg in self.value_pool.get(*args) {
                            visit(errs, arg);
                        }
                    }
                    ValueKind::Project { call, .. } => visit(errs, *call),
                    ValueKind::Gamma {
                        condition, inputs, ..
                    } => {
                        visit(errs, *condition);
                        for &input in self.value_pool.get(*inputs) {
                            visit(errs, input);
                        }
                    }
                    ValueKind::Theta {
                        loop_vars,
                        condition,
                        region_id,
                        ..
                    } => {
                        for &var in self.value_pool.get(*loop_vars) {
                            visit(errs, var);
                        }
                        // The repetition predicate lives inside the body
                        // region: check it there, positionlessly (it is
                        // consumed at the body's end).
                        let body = region_id.0;
                        match owner[condition.0 as usize] {
                            Owner::Node { region, .. } if region == body => {}
                            Owner::Param { region } if region == body => {}
                            _ if region_free(*condition) => {}
                            _ => errs.push(RVSDGVerificationError::ValueUsedOutOfScope {
                                user,
                                operand: *condition,
                                region: *region_id,
                            }),
                        }
                    }
                    ValueKind::RegionResult { values, .. } => {
                        // Structural: its span IS the owning region's
                        // results, checked in the results loop below.
                        let _ = values;
                    }
                    ValueKind::Const(_)
                    | ValueKind::ConstPoolRef(_)
                    | ValueKind::GlobalRef(_)
                    | ValueKind::FuncAddr(_)
                    | ValueKind::Fence { .. }
                    | ValueKind::Lambda { .. }
                    | ValueKind::Phi { .. }
                    | ValueKind::RegionParam { .. } => {}
                }
            }

            // A region's results must be visible inside it (at its end):
            // one of its nodes, one of its parameters, or region-free.
            for &result in self.value_pool.get(region.results) {
                let visible = region_free(result)
                    || matches!(
                        owner[result.0 as usize],
                        Owner::Node { region, .. } | Owner::Param { region }
                            if region == user_region
                    );
                if !visible {
                    errs.push(RVSDGVerificationError::ResultNotInRegion {
                        region: RegionId(user_region),
                        operand: result,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ConstValue, Linkage, RVSDGMod,
        builder::BranchResult,
        func::FnResult,
        types::{BOOL, I32},
        verify::RVSDGVerificationError,
    };

    /// A gamma arm whose result reaches directly into the enclosing region
    /// (the closure captures an outer ValueId instead of taking a
    /// parameter) -- the exact bug class this pass exists to catch.
    #[test]
    fn arm_reaching_into_enclosing_region_is_caught() {
        let mut rvsdg = RVSDGMod::new_host(String::from("test"));
        let main_fn = rvsdg.declare_fn(String::from("main"), &[], &[I32], Linkage::Internal);
        rvsdg
            .define_fn(main_fn, |rb, state| {
                let outer = rb.const_i32(7);
                let outer_sum = rb.binary(
                    crate::rvsdg::BinaryOp::Add,
                    Default::default(),
                    outer,
                    outer,
                    I32,
                );
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let res = rb.gamma(
                    predicate,
                    state,
                    &[],
                    |_rb| {
                        // Violation: uses the enclosing region's node.
                        Ok(BranchResult {
                            state,
                            values: vec![outer_sum],
                        })
                    },
                    |rb| {
                        let zero = rb.const_i32(0);
                        Ok(BranchResult {
                            state,
                            values: vec![zero],
                        })
                    },
                )?;
                Ok(FnResult {
                    state: res.state,
                    values: vec![res.result(0)],
                })
            })
            .unwrap();

        let errs = rvsdg.verify();
        assert!(
            errs.iter().any(|e| matches!(
                e,
                RVSDGVerificationError::ResultNotInRegion { .. }
                    | RVSDGVerificationError::ValueUsedOutOfScope { .. }
            )),
            "expected a scope violation, got: {errs:?}"
        );
    }
}
