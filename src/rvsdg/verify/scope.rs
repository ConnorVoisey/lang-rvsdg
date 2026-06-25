use crate::rvsdg::{RVSDGMod, Region, RegionId, ValueId, verify::RVSDGVerificationError};

impl RVSDGMod {
    pub(super) fn verify_scope(&self, errs: &mut Vec<RVSDGVerificationError>) {
        for (i, region) in self.regions.iter().enumerate() {
            let region_id = RegionId(i as u32);
            for val_id in region.nodes.iter() {
                self.verify_val_in_region(errs, region, region_id, *val_id);
            }
        }
    }

    #[inline(always)]
    fn verify_val_in_region(
        &self,
        errs: &mut Vec<RVSDGVerificationError>,
        region: &Region,
        region_id: RegionId,
        val_id: ValueId,
    ) {
        if !region.nodes.contains(&val_id) {
            errs.push(RVSDGVerificationError::ValueUsedOutOfScope(
                val_id, region_id,
            ));
        };
    }
}
