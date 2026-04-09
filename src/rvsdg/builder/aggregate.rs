use crate::rvsdg::{Value, ValueId, ValueKind, types::TypeRef};

use super::RegionBuilder;

impl<'a> RegionBuilder<'a> {
    #[inline]
    pub fn extract_lane(
        &mut self,
        vector: ValueId,
        index: ValueId,
        element_type: TypeRef,
    ) -> ValueId {
        self.add_value(Value {
            ty: element_type,
            kind: ValueKind::ExtractLane { vector, index },
        })
    }

    #[inline]
    pub fn insert_lane(
        &mut self,
        vector: ValueId,
        index: ValueId,
        value: ValueId,
        vector_type: TypeRef,
    ) -> ValueId {
        self.add_value(Value {
            ty: vector_type,
            kind: ValueKind::InsertLane {
                vector,
                index,
                value,
            },
        })
    }

    #[inline]
    pub fn shuffle_lanes(
        &mut self,
        left: ValueId,
        right: ValueId,
        mask: &[ValueId],
        result_type: TypeRef,
    ) -> ValueId {
        let mask_span = self.graph.value_pool.push_slice(mask);
        self.add_value(Value {
            ty: result_type,
            kind: ValueKind::ShuffleLanes {
                left,
                right,
                mask: mask_span,
            },
        })
    }

    #[inline]
    pub fn extract_field(
        &mut self,
        aggregate: ValueId,
        indices: &[u32],
        field_type: TypeRef,
    ) -> ValueId {
        let indices_span = self.graph.u32_pool.push_slice(indices);
        self.add_value(Value {
            ty: field_type,
            kind: ValueKind::ExtractField {
                aggregate,
                indices: indices_span,
            },
        })
    }

    #[inline]
    pub fn insert_field(
        &mut self,
        aggregate: ValueId,
        value: ValueId,
        indices: &[u32],
        aggregate_type: TypeRef,
    ) -> ValueId {
        let indices_span = self.graph.u32_pool.push_slice(indices);
        self.add_value(Value {
            ty: aggregate_type,
            kind: ValueKind::InsertField {
                aggregate,
                value,
                indices: indices_span,
            },
        })
    }

    #[inline]
    pub fn ptr_offset(
        &mut self,
        base: ValueId,
        base_type: TypeRef,
        indices: &[ValueId],
        result_type: TypeRef,
        inbounds: bool,
    ) -> ValueId {
        let indices_span = self.graph.value_pool.push_slice(indices);
        self.add_value(Value {
            ty: result_type,
            kind: ValueKind::PtrOffset {
                base,
                base_type,
                indices: indices_span,
                inbounds,
            },
        })
    }
}
