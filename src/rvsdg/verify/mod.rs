use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use thiserror::Error;

use crate::rvsdg::{
    ConstId, FuncId, GlobalId, RVSDGMod, RegionId, ValueId, function_graph::FunctionGraph,
    module_tables::ModuleTables,
};

pub mod ids;
pub mod ownership;
pub mod predicate_form;
pub mod scope;
pub mod state;
pub mod typing;

impl RVSDGMod {
    #[tracing::instrument(skip_all)]
    pub fn verify(&self) -> Vec<RVSDGVerificationError> {
        self.graphs
            .par_iter()
            .map(|func| match func {
                Some(func) => func.verify(&self.tables),
                None => vec![],
            })
            // inefficient cloning, but this is the unhappy path
            .reduce(|| vec![], |a, b| [a, b].concat())
    }
}
impl FunctionGraph {
    #[tracing::instrument(skip_all)]
    pub fn verify(&self, module_tables: &ModuleTables) -> Vec<RVSDGVerificationError> {
        let mut errs = vec![];
        // Structural precondition, checked before anything else: every
        // pass below reads region lists through the sealed pool handles,
        // so an unsealed region would panic on a bogus slice instead of
        // reporting. Like the other finaliser obligations (exit state,
        // owner), a missed seal is a verification error, not a crash.
        for (index, region) in self.regions.iter().enumerate() {
            if !region.is_sealed() {
                errs.push(RVSDGVerificationError::RegionUnsealed(RegionId(
                    index as u32,
                )));
            }
        }
        if !errs.is_empty() {
            return errs;
        }
        self.verify_ids(module_tables, &mut errs);
        // Id validity is a precondition for every pass below: they index
        // the value arrays with operand ids, so an out-of-range id would
        // panic there instead of being reported.
        if !errs.is_empty() {
            return errs;
        }
        let ownership = self.build_value_ownership(&mut errs);
        self.verify_scope(&ownership, &mut errs);
        self.verify_state(&ownership, &mut errs);
        self.verify_chain_continuity(&mut errs);
        self.verify_typing(&mut errs);
        self.verify_region_ownership(&mut errs);
        self.verify_predicate_form(&mut errs);
        errs
    }
}

#[derive(Error, Debug, Clone)]
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

    #[error("region {0} was never sealed")]
    RegionUnsealed(RegionId),

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

    #[error("state producer {value} has type {ty:?}, expected {expected}")]
    StateProducerTypeWrong {
        value: ValueId,
        ty: crate::rvsdg::types::TypeRef,
        expected: &'static str,
    },

    #[error("value {0} is State-typed but is not a state producer or a state tail parameter")]
    StateTypedNonProducer(ValueId),

    #[error("state merge {merge} joins input {input}, which is not a read of its alias class")]
    StateMergeClassMismatch { merge: ValueId, input: ValueId },

    #[error(
        "state-typed value {value} appears in region {region}'s value interface; state crosses \
         region boundaries only through the state tails"
    )]
    StateTypedRegionInterface { region: RegionId, value: ValueId },

    #[error(
        "state-typed value {operand} flows into a data operand of {user}; state travels only \
         through state operands and region state tails"
    )]
    StateTypedDataOperand { user: ValueId, operand: ValueId },

    #[error(
        "region {region}'s state {side} tail has {len} entries; construction threads exactly \
         two chains (memory, io)"
    )]
    StateTailWrongArity {
        region: RegionId,
        side: &'static str,
        len: usize,
    },

    #[error("region {region}'s state tail entry {value} is not a value of the {chain} chain")]
    StateTailWrongChain {
        region: RegionId,
        value: ValueId,
        chain: &'static str,
    },

    #[error(
        "construct {construct}'s projections are malformed: expected data projections, then at \
         most one memory state projection, then at most one io projection"
    )]
    ConstructStateProjectionsMalformed { construct: ValueId },

    #[error(
        "construct {construct}'s subregion {region} carries entry state tails that differ from \
         its first subregion's; a construct's chain inputs must be identical across its \
         subregions"
    )]
    ConstructEntryTailsDisagree {
        construct: ValueId,
        region: RegionId,
    },

    #[error(
        "effectful state value {value} is not reachable from any region's {chain} exit; the \
         chain skips its effect, so ordering is lost and elimination would delete it"
    )]
    StateEffectUnrooted { value: ValueId, chain: &'static str },

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
