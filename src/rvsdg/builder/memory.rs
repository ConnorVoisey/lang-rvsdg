use crate::rvsdg::{
    AliasClassId, AtomicRMWOp, MemoryOrdering, StateKind, Value, ValueId, ValueKind,
    types::{BOOL, TypeRef},
};

use super::{CompareAndSwapResult, RegionBuilder};

// State threading is internal: every op pulls its input state from the
// region's scratch registers (reads fan out from the chain's newest
// write; writes flatten pending reads behind them), so ops take and
// return data values only.

impl<'a> RegionBuilder<'a> {
    /// Returns the loaded value.
    #[inline]
    pub fn load(
        &mut self,
        addr: ValueId,
        loaded_type: TypeRef,
        align: Option<u32>,
        volatile: bool,
    ) -> ValueId {
        let input_state = self.graph.state_current(self.region_id);
        let load_val = self.add_value(Value {
            ty: TypeRef::State(StateKind::MemoryRead(AliasClassId(0))),
            kind: ValueKind::Load {
                state: input_state,
                addr,
                loaded_type,
                align,
                volatile,
            },
        });
        self.graph.state_read(self.region_id, load_val);
        self.add_value(Value {
            ty: loaded_type,
            kind: ValueKind::Project {
                call: load_val,
                index: 0,
            },
        })
    }

    #[inline]
    pub fn store(&mut self, addr: ValueId, value: ValueId, align: Option<u32>, volatile: bool) {
        self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::Store {
                state,
                addr,
                value,
                align,
                volatile,
            },
        });
    }

    /// Returns the allocated pointer.
    #[inline]
    pub fn alloca(
        &mut self,
        elem_type: TypeRef,
        count: ValueId,
        ptr_type: TypeRef,
        align: Option<u32>,
    ) -> ValueId {
        let alloca_state = self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::Alloca {
                state,
                elem_type,
                count,
                align,
            },
        });
        self.add_value(Value {
            ty: ptr_type,
            kind: ValueKind::Project {
                call: alloca_state.0,
                index: 0,
            },
        })
    }

    /// Returns the loaded value.
    #[inline]
    pub fn atomic_load(
        &mut self,
        addr: ValueId,
        loaded_type: TypeRef,
        ordering: MemoryOrdering,
        align: Option<u32>,
        volatile: bool,
    ) -> ValueId {
        // Atomic loads thread as writes: their ordering constraint must
        // hold every later op behind them, which read fan-out would lose.
        let load_state = self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::AtomicLoad {
                state,
                addr,
                loaded_type,
                ordering,
                align,
                volatile,
            },
        });
        self.add_value(Value {
            ty: loaded_type,
            kind: ValueKind::Project {
                call: load_state.0,
                index: 0,
            },
        })
    }

    #[inline]
    pub fn atomic_store(
        &mut self,
        addr: ValueId,
        value: ValueId,
        ordering: MemoryOrdering,
        align: Option<u32>,
        volatile: bool,
    ) {
        self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::AtomicStore {
                state,
                addr,
                value,
                ordering,
                align,
                volatile,
            },
        });
    }

    /// Returns the old value.
    #[inline]
    pub fn atomic_read_modify_write(
        &mut self,
        addr: ValueId,
        value: ValueId,
        op: AtomicRMWOp,
        ordering: MemoryOrdering,
        value_type: TypeRef,
        volatile: bool,
    ) -> ValueId {
        let rmw_state = self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::AtomicReadModifyWrite {
                state,
                addr,
                value,
                op,
                ordering,
                volatile,
            },
        });
        self.add_value(Value {
            ty: value_type,
            kind: ValueKind::Project {
                call: rmw_state.0,
                index: 0,
            },
        })
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn compare_and_swap(
        &mut self,
        addr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
        value_type: TypeRef,
        volatile: bool,
    ) -> CompareAndSwapResult {
        let cas_state = self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::CompareAndSwap {
                state,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
                volatile,
            },
        });
        let old_value = self.add_value(Value {
            ty: value_type,
            kind: ValueKind::Project {
                call: cas_state.0,
                index: 0,
            },
        });
        let success = self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::Project {
                call: cas_state.0,
                index: 1,
            },
        });
        CompareAndSwapResult {
            node: cas_state.0,
            old_value,
            success,
        }
    }

    #[inline]
    pub fn fence(&mut self, ordering: MemoryOrdering) {
        self.graph.state_write(self.region_id, |state| Value {
            ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
            kind: ValueKind::Fence { state, ordering },
        });
    }

    #[inline]
    pub fn freeze(&mut self, value: ValueId, ty: TypeRef) -> ValueId {
        self.add_value(Value {
            ty,
            kind: ValueKind::Freeze { value },
        })
    }
}
