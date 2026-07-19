use thiserror::Error;

use crate::rvsdg::{ConstId, FuncId, GlobalId, RVSDGMod, RegionId, ValueId};

pub mod ids;
pub mod ownership;
pub mod predicate_form;
pub mod scope;
pub mod state;

impl RVSDGMod {
    #[tracing::instrument(skip_all)]
    pub fn verify(&self) -> Vec<RVSDGVerificationError> {
        let mut errs = vec![];
        self.verify_ids(&mut errs);
        let ownership = self.build_value_ownership(&mut errs);
        self.verify_scope(&ownership, &mut errs);
        self.verify_state(&ownership, &mut errs);
        self.verify_region_ownership(&mut errs);
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

    #[error(
        "value {operand} is used by {user} in region {region}, but is neither a node of that \
         region, one of its parameters, nor a region-free constant"
    )]
    ValueUsedOutOfScope {
        user: ValueId,
        operand: ValueId,
        region: RegionId,
    },

    #[error("value {operand} is used by {user} before its definition in the region")]
    ValueUsedBeforeDefinition { user: ValueId, operand: ValueId },

    #[error("value {0} appears in the node list of more than one region")]
    ValueInMultipleRegions(ValueId),

    #[error("result {operand} of region {region} is not visible inside that region")]
    ResultNotInRegion { region: RegionId, operand: ValueId },

    #[error(
        "state edge into {user}: operand {operand} is neither region {region}'s entry state nor \
         an earlier state-producing node of that region"
    )]
    StateEdgeOutOfScope {
        user: ValueId,
        operand: ValueId,
        region: RegionId,
    },

    #[error("state edge into {user}: operand {operand} is not a state-producing node")]
    StateEdgeFromNonStateNode { user: ValueId, operand: ValueId },

    #[error("state edge into {user}: operand {operand} is defined later in the same region")]
    StateEdgeUsedBeforeDefinition { user: ValueId, operand: ValueId },

    #[error(
        "region {region}'s exit state {operand} is neither its entry state nor a \
         state-producing node of that region"
    )]
    RegionExitStateInvalid { region: RegionId, operand: ValueId },

    #[error("region {0}'s exit state was never set by its finaliser")]
    RegionExitStateUnset(RegionId),

    #[error("region {0}'s owner was never set by its finaliser")]
    RegionOwnerUnset(RegionId),

    #[error("region {region}'s owner {owner} is not a construct whose region list names it")]
    RegionOwnerInvalid { region: RegionId, owner: ValueId },

    #[error(
        "region {region}'s parameter list holds {param}, which is not a RegionParam naming \
         that region back"
    )]
    RegionParamLinkInvalid { region: RegionId, param: ValueId },

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
