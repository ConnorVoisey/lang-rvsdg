//! **Region ownership** -- the back links from regions to the constructs
//! that own them. `Region::owner` cannot be set when the region is
//! created (arms are emitted before their gamma value exists), so like
//! `exit_state` it is stamped by the construct's finaliser; and
//! `RegionParam::region` is stamped at parameter creation. Passes
//! navigate interfaces through these links (dead node elimination maps
//! an arm parameter to the gamma input slot feeding it), so a stale or
//! missing link silently redirects a pass to the wrong construct. This
//! pass holds both links to the forward structure:
//!
//! - every region's `owner` is set, and that construct's region list
//!   names the region back;
//! - every value in a region's `params` list is a `RegionParam` whose
//!   `region` field names that region back.

use crate::rvsdg::{RVSDGMod, RegionId, ValueId, ValueKind, verify::RVSDGVerificationError};

impl RVSDGMod {
    pub(super) fn verify_region_ownership(&self, errs: &mut Vec<RVSDGVerificationError>) {
        for (region_index, region) in self.regions.iter().enumerate() {
            let region_id = RegionId(region_index as u32);

            if region.owner == ValueId::INVALID {
                errs.push(RVSDGVerificationError::RegionOwnerUnset(region_id));
            } else {
                let names_region_back = match self.get_value_kind(region.owner) {
                    ValueKind::Lambda { region: r, .. } | ValueKind::Phi { region: r, .. } => {
                        *r == region_id
                    }
                    ValueKind::Theta { region_id: r, .. } => *r == region_id,
                    ValueKind::Gamma { regions, .. } => {
                        self.region_pool.get(*regions).contains(&region_id)
                    }
                    _ => false,
                };
                if !names_region_back {
                    errs.push(RVSDGVerificationError::RegionOwnerInvalid {
                        region: region_id,
                        owner: region.owner,
                    });
                }
            }

            for &param in &region.params {
                let names_region_back = matches!(
                    self.get_value_kind(param),
                    ValueKind::RegionParam { region: r, .. } if *r == region_id
                );
                if !names_region_back {
                    errs.push(RVSDGVerificationError::RegionParamLinkInvalid {
                        region: region_id,
                        param,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ConstValue, ICmpPred, Linkage, RVSDGMod, RegionId, ValueId,
        ValueKind,
        builder::{BranchResult, LoopResult},
        func::FnResult,
        types::{BOOL, I32},
        verify::RVSDGVerificationError,
    };

    /// One function containing a gamma and a theta: every region's owner
    /// names its construct and every parameter names its region, so the
    /// whole graph verifies and the links read back correctly.
    fn build_gamma_theta_module() -> RVSDGMod {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb, state| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let picked = rb.gamma(
                predicate,
                state,
                &[x, y],
                |rb| {
                    let a = rb.param(0);
                    Ok(BranchResult {
                        state,
                        values: vec![a],
                    })
                },
                |rb| {
                    let b = rb.param(1);
                    Ok(BranchResult {
                        state,
                        values: vec![b],
                    })
                },
            )?;
            let looped = rb.theta(picked.state, &[picked.result(0)], |rb| {
                let i = rb.param(0);
                let one = rb.const_i32(1);
                let next_i = rb.binary(BinaryOp::Add, ArithFlags::default(), i, one, I32);
                let five = rb.const_i32(5);
                let condition = rb.icmp(ICmpPred::SignedLt, next_i, five);
                // The body is pure: its chain ends on its entry state,
                // which is the state the theta was created with.
                Ok(LoopResult {
                    condition,
                    next_state: picked.state,
                    next_vars: vec![next_i],
                })
            })?;
            Ok(FnResult {
                state: looped.state,
                values: vec![looped.result(0)],
            })
        })
        .unwrap();
        m
    }

    #[test]
    fn construction_stamps_every_owner_and_param_link() {
        let m = build_gamma_theta_module();
        let errs = m.verify();
        assert!(errs.is_empty(), "expected clean graph, got: {errs:?}");

        // Spot-check the links resolve to the right constructs.
        let lambda = m.functions[0].lambda_val.unwrap();
        let ValueKind::Lambda { region, .. } = &m.get_value_kind(lambda) else {
            unreachable!();
        };
        assert_eq!(m.regions[region.0 as usize].owner, lambda);
        for &param in &m.regions[region.0 as usize].params {
            let ValueKind::RegionParam { region: r, .. } = &m.get_value_kind(param) else {
                unreachable!();
            };
            assert_eq!(*r, *region);
        }
    }

    #[test]
    fn unset_owner_is_caught() {
        let mut m = build_gamma_theta_module();
        m.regions[0].owner = ValueId::INVALID;
        let errs = m.verify();
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionOwnerUnset(_))),
            "expected an unset-owner error, got: {errs:?}"
        );
    }

    #[test]
    fn owner_not_naming_region_back_is_caught() {
        let mut m = build_gamma_theta_module();
        // Point the function region's owner at a value that is not a
        // construct owning it (the returned Add).
        let add = m
            .regions
            .iter()
            .flat_map(|r| r.nodes.iter())
            .find(|id| matches!(m.get_value_kind(**id), ValueKind::Binary { .. }))
            .copied()
            .unwrap();
        m.regions[0].owner = add;
        let errs = m.verify();
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionOwnerInvalid { .. })),
            "expected an invalid-owner error, got: {errs:?}"
        );
    }

    #[test]
    fn param_naming_wrong_region_is_caught() {
        let mut m = build_gamma_theta_module();
        let param = m.regions[0].params[0];
        let ValueKind::RegionParam { region, .. } = &mut m.value_kinds[param.0 as usize] else {
            unreachable!();
        };
        *region = RegionId(region.0 + 1);
        let errs = m.verify();
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::RegionParamLinkInvalid { .. })),
            "expected a param-link error, got: {errs:?}"
        );
    }
}
