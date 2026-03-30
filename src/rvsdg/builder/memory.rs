use crate::rvsdg::{
    AtomicRMWOp, MemoryOrdering, State, Value, ValueId, ValueKind,
    types::{BOOL, TypeRef},
};

use super::{AllocaResult, CompareAndSwapResult, LoadResult, RegionBuilder};

impl<'a> RegionBuilder<'a> {
    #[inline]
    pub fn load(&mut self, state: State, addr: ValueId, loaded_type: TypeRef) -> LoadResult {
        let load_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Load {
                state,
                addr,
                loaded_type,
            },
        });
        let value = self.add_value(Value {
            ty: loaded_type,
            kind: ValueKind::Project {
                call: load_val,
                index: 0,
            },
        });
        LoadResult {
            state: State(load_val),
            value,
        }
    }

    #[inline]
    pub fn store(&mut self, state: State, addr: ValueId, value: ValueId) -> State {
        let store_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Store { state, addr, value },
        });
        State(store_val)
    }

    #[inline]
    pub fn alloca(
        &mut self,
        state: State,
        elem_type: TypeRef,
        count: ValueId,
        ptr_type: TypeRef,
    ) -> AllocaResult {
        let alloca_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Alloca {
                state,
                elem_type,
                count,
            },
        });
        let ptr = self.add_value(Value {
            ty: ptr_type,
            kind: ValueKind::Project {
                call: alloca_val,
                index: 0,
            },
        });
        AllocaResult {
            state: State(alloca_val),
            ptr,
        }
    }

    #[inline]
    pub fn atomic_load(
        &mut self,
        state: State,
        addr: ValueId,
        loaded_type: TypeRef,
        ordering: MemoryOrdering,
    ) -> LoadResult {
        let load_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::AtomicLoad {
                state,
                addr,
                loaded_type,
                ordering,
            },
        });
        let value = self.add_value(Value {
            ty: loaded_type,
            kind: ValueKind::Project {
                call: load_val,
                index: 0,
            },
        });
        LoadResult {
            state: State(load_val),
            value,
        }
    }

    #[inline]
    pub fn atomic_store(
        &mut self,
        state: State,
        addr: ValueId,
        value: ValueId,
        ordering: MemoryOrdering,
    ) -> State {
        let val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::AtomicStore {
                state,
                addr,
                value,
                ordering,
            },
        });
        State(val)
    }

    #[inline]
    pub fn atomic_read_modify_write(
        &mut self,
        state: State,
        addr: ValueId,
        value: ValueId,
        op: AtomicRMWOp,
        ordering: MemoryOrdering,
        value_type: TypeRef,
    ) -> LoadResult {
        let rmw_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::AtomicReadModifyWrite {
                state,
                addr,
                value,
                op,
                ordering,
            },
        });
        let old_value = self.add_value(Value {
            ty: value_type,
            kind: ValueKind::Project {
                call: rmw_val,
                index: 0,
            },
        });
        LoadResult {
            state: State(rmw_val),
            value: old_value,
        }
    }

    #[inline]
    pub fn compare_and_swap(
        &mut self,
        state: State,
        addr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
        value_type: TypeRef,
    ) -> CompareAndSwapResult {
        let cas_val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::CompareAndSwap {
                state,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            },
        });
        let old_value = self.add_value(Value {
            ty: value_type,
            kind: ValueKind::Project {
                call: cas_val,
                index: 0,
            },
        });
        let success = self.add_value(Value {
            ty: BOOL,
            kind: ValueKind::Project {
                call: cas_val,
                index: 1,
            },
        });
        CompareAndSwapResult {
            state: State(cas_val),
            old_value,
            success,
        }
    }

    #[inline]
    pub fn fence(&mut self, state: State, ordering: MemoryOrdering) -> State {
        let val = self.add_value(Value {
            ty: TypeRef::State,
            kind: ValueKind::Fence { state, ordering },
        });
        State(val)
    }

    #[inline]
    pub fn freeze(&mut self, value: ValueId, ty: TypeRef) -> ValueId {
        self.add_value(Value {
            ty,
            kind: ValueKind::Freeze { value },
        })
    }
}
