use thiserror::Error;

use crate::rvsdg::{ConstId, FuncId, GlobalId, RVSDGMod, RegionId, ValueId};

pub mod ids;
pub mod predicate_form;
pub mod scope;

impl RVSDGMod {
    pub fn verify(&self) -> Vec<RVSDGVerificationError> {
        let mut errs = vec![];
        self.verify_ids(&mut errs);
        self.verify_scope(&mut errs);
        self.verify_predicate_form(&mut errs);
        errs
    }
}

#[derive(Error, Debug)]
pub enum RVSDGVerificationError {
    #[error("invalid value id {0}")]
    InvalidValueId(ValueId),

    #[error("invalid function id {0}")]
    InvalidFnId(FuncId),

    #[error("invalid const id {0}")]
    InvalidConstId(ConstId),

    #[error("invalid global id {0}")]
    InvalidGlobalId(GlobalId),

    #[error("invalid region id {0}")]
    InvalidRegionId(RegionId),

    #[error("Value {0}, is used in region {1}, but it isn't declared within this region")]
    ValueUsedOutOfScope(ValueId, RegionId),

    #[error("Region {region_id} takes in {input_count} params yet returns {output_count} params")]
    RegionInvalidArgReturnCount {
        region_id: RegionId,
        input_count: usize,
        output_count: usize,
    },

    #[error(
        "predicate {0} flows into an ordinary operand slot; predicates may only feed a gamma \
         decision, a theta repetition predicate, or a region result (predicate continuation form)"
    )]
    PredicateNonConditionUse(ValueId),

    #[error(
        "predicate {0} has {1} uses; predicate continuation form allows at most one, so that \
         control flow reconstruction can trace each predicate to its single consumer"
    )]
    PredicateUsedMoreThanOnce(ValueId, u32),
}
