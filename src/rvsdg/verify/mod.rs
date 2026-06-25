use thiserror::Error;

use crate::rvsdg::{ConstId, FuncId, GlobalId, RVSDGMod, RegionId, ValueId};

pub mod ids;
pub mod scope;

impl RVSDGMod {
    pub fn verify(&self) -> Vec<RVSDGVerificationError> {
        let mut errs = vec![];
        self.verify_ids(&mut errs);
        self.verify_scope(&mut errs);
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
}
