use crate::rvsdg::{
    AliasClassId, AtomicRMWOp, MemoryOrdering, StateKind, Value, ValueId, ValueKind,
    memory_alias::origin::MemoryOrigin,
    types::{BOOL, TypeRef},
};

use super::{CompareAndSwapResult, RegionBuilder};

// State threading is internal: every op pulls its input state from the
// region's scratch registers (reads fan out from the chain's newest
// write; writes flatten pending reads behind them), so ops take and
// return data values only.
//
// Memory origins: op NODES carry None (an op is classified at
// resolution by chasing its addr operand's tag, never its own); only
// pointer-typed RESULTS carry tags. A value loaded from memory has
// Unknown origin -- its pointee could be anything -- and a non-pointer
// result carries None (MemoryOrigin::unknown_if_ptr).

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
        let load_val = self.add_value(
            Value {
                ty: TypeRef::State(StateKind::MemoryRead(AliasClassId(0))),
                kind: ValueKind::Load {
                    state: input_state,
                    addr,
                    loaded_type,
                    align,
                    volatile,
                },
            },
            MemoryOrigin::None,
        );
        self.graph.record_access_event(load_val);
        self.graph.state_read(self.region_id, load_val);
        self.add_value(
            Value {
                ty: loaded_type,
                kind: ValueKind::Project {
                    call: load_val,
                    index: 0,
                },
            },
            MemoryOrigin::unknown_if_ptr(loaded_type),
        )
    }

    #[inline]
    pub fn store(&mut self, addr: ValueId, value: ValueId, align: Option<u32>, volatile: bool) {
        let op = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::Store {
                        state,
                        addr,
                        value,
                        align,
                        volatile,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(op.0);
        self.graph.record_escape_event(value);
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
        // The node id is only known AFTER state_write: flattening
        // pending reads may push a StateMerge first, so a
        // value_kinds.len() taken here would name the merge instead.
        let alloca_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::Alloca {
                        state,
                        elem_type,
                        count,
                        align,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_alloca_event(alloca_state.0);
        self.add_value(
            Value {
                ty: ptr_type,
                kind: ValueKind::Project {
                    call: alloca_state.0,
                    index: 0,
                },
            },
            MemoryOrigin::Alloca(alloca_state.0),
        )
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
        let load_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::AtomicLoad {
                        state,
                        addr,
                        loaded_type,
                        ordering,
                        align,
                        volatile,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(load_state.0);
        self.add_value(
            Value {
                ty: loaded_type,
                kind: ValueKind::Project {
                    call: load_state.0,
                    index: 0,
                },
            },
            MemoryOrigin::unknown_if_ptr(loaded_type),
        )
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
        let op = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::AtomicStore {
                        state,
                        addr,
                        value,
                        ordering,
                        align,
                        volatile,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(op.0);
        self.graph.record_escape_event(value);
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
        let rmw_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::AtomicReadModifyWrite {
                        state,
                        addr,
                        value,
                        op,
                        ordering,
                        volatile,
                    },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(rmw_state.0);
        // An xchg of a pointer publishes it (the old value may be read
        // back by anyone); the pointer gate skips integer RMWs.
        self.graph.record_escape_event(value);
        self.add_value(
            Value {
                ty: value_type,
                kind: ValueKind::Project {
                    call: rmw_state.0,
                    index: 0,
                },
            },
            MemoryOrigin::unknown_if_ptr(value_type),
        )
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
        let cas_state = self.graph.state_write(self.region_id, |state| {
            (
                Value {
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
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(cas_state.0);
        // Both the compared-against and the stored pointer end up
        // reachable through the cell.
        self.graph.record_escape_event(expected);
        self.graph.record_escape_event(desired);
        let old_value = self.add_value(
            Value {
                ty: value_type,
                kind: ValueKind::Project {
                    call: cas_state.0,
                    index: 0,
                },
            },
            MemoryOrigin::unknown_if_ptr(value_type),
        );
        let success = self.add_value(
            Value {
                ty: BOOL,
                kind: ValueKind::Project {
                    call: cas_state.0,
                    index: 1,
                },
            },
            MemoryOrigin::None,
        );
        CompareAndSwapResult {
            node: cas_state.0,
            old_value,
            success,
        }
    }

    #[inline]
    pub fn fence(&mut self, ordering: MemoryOrdering) {
        let op = self.graph.state_write(self.region_id, |state| {
            (
                Value {
                    ty: TypeRef::State(StateKind::MemoryWrite(AliasClassId(0))),
                    kind: ValueKind::Fence { state, ordering },
                },
                MemoryOrigin::None,
            )
        });
        self.graph.record_access_event(op.0);
    }

    #[inline]
    pub fn freeze(&mut self, value: ValueId, ty: TypeRef) -> ValueId {
        // A frozen pointer is the same pointer: link, never copy (the
        // base may widen later).
        let origin = if matches!(ty, TypeRef::Ptr(_)) {
            MemoryOrigin::Derived(value)
        } else {
            MemoryOrigin::None
        };
        self.add_value(
            Value {
                ty,
                kind: ValueKind::Freeze { value },
            },
            origin,
        )
    }
}
