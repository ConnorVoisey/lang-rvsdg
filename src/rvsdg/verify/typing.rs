//! **State typing** -- the state chains are typed edges, and the types
//! are load-bearing for the passes that come after construction: the
//! alias-class split keys on `State(MemoryRead/MemoryWrite(class))`, io
//! ordering on `State(InputOutput)`, and purity derivation on tail
//! comparison. The scope and state passes hold the EDGES to their
//! structural rules; this pass holds the VALUES to their types:
//!
//! - kind -> type: a single-chain producer carries its chain's type
//!   (loads and merges are read evidence; writes, constructs and the
//!   rest of the memory ops are writes). Calls stay Void-typed: they
//!   advance several chains at once, so no single State type fits, and
//!   consumers identify them by kind.
//! - type -> kind: a State-typed value must be a producer or a state
//!   tail parameter; State never types ordinary computation.
//! - a merge joins reads of one alias class.
//! - State-typed values stay out of the data graph: never an ordinary
//!   operand, a value parameter, or a value result. Projections are the
//!   sanctioned data-out-of-state-node edge and are exempt.
//! - every region's state tails carry one well-shaped entry per chain,
//!   in [memory, io] order.

use crate::rvsdg::{
    RegionId, StateKind, ValueId, ValueKind, function_graph::FunctionGraph, types::TypeRef,
    verify::RVSDGVerificationError,
};

impl FunctionGraph {
    pub(super) fn verify_typing(&self, errs: &mut Vec<RVSDGVerificationError>) {
        let state_kind_of = |value: ValueId| match self.get_value_type(value) {
            TypeRef::State(kind) => Some(*kind),
            _ => None,
        };
        // Legal occupants of a memory-chain slot: memory-typed values,
        // or a call (multi-chain, identified by kind).
        let memory_shaped = |value: ValueId| {
            matches!(
                state_kind_of(value),
                Some(StateKind::MemoryRead(_) | StateKind::MemoryWrite(_))
            ) || self.get_value_kind(value).is_call()
        };
        // Legal occupants of an io-chain slot: io-typed values (entry
        // params, constructs' io state projections), or a call.
        let io_shaped = |value: ValueId| {
            matches!(state_kind_of(value), Some(StateKind::InputOutput))
                || self.get_value_kind(value).is_call()
        };

        for index in 0..self.value_kinds.len() {
            let value = ValueId(index as u32);
            let kind = self.get_value_kind(value);
            let state_ty = state_kind_of(value);

            // Kind -> type. (Calls and constructs stay Void: a call
            // advances several chains at once so no single State type
            // fits, and a construct's chains live on its projections.)
            let expected = match kind {
                ValueKind::Load { .. } | ValueKind::StateMerge { .. } => {
                    (!matches!(state_ty, Some(StateKind::MemoryRead(_))))
                        .then_some("State(MemoryRead)")
                }
                writer if writer.is_memory_op() => {
                    (!matches!(state_ty, Some(StateKind::MemoryWrite(_))))
                        .then_some("State(MemoryWrite)")
                }
                // Only a construct carries state projections, typed by
                // the chain they continue ([memory write, io]; reads
                // never). Their ORDER among the construct's projections
                // is checked per construct in the region sweep below;
                // position is not fixed, since a bypassed chain's
                // projection is removable.
                ValueKind::Project { call, .. } => match state_ty {
                    None => None,
                    Some(kind) => {
                        let legal = self.get_value_kind(*call).is_construct()
                            && matches!(kind, StateKind::MemoryWrite(_) | StateKind::InputOutput);
                        (!legal).then_some(
                            "a data type (only constructs carry State(MemoryWrite)/State(InputOutput) projections)",
                        )
                    }
                },
                _ => None,
            };
            if let Some(expected) = expected {
                errs.push(RVSDGVerificationError::StateProducerTypeWrong {
                    value,
                    ty: *self.get_value_type(value),
                    expected,
                });
            }

            // Type -> kind. RegionParams are legal carriers (the tails);
            // a State-typed VALUE param is caught by the interface scan
            // below, since the params list is what defines that role.
            // Projections are legal carriers too (construct state
            // projections); their own rule above pins which ones.
            if state_ty.is_some()
                && !kind.is_memory_op()
                && !matches!(
                    kind,
                    ValueKind::StateMerge { .. }
                        | ValueKind::RegionParam { .. }
                        | ValueKind::Project { .. }
                )
            {
                errs.push(RVSDGVerificationError::StateTypedNonProducer(value));
            }

            // A merge joins reads of one alias class -- its own.
            if let ValueKind::StateMerge { inputs } = kind
                && let Some(StateKind::MemoryRead(merge_class)) = state_ty
            {
                for &input in self.value_pool.get(*inputs) {
                    let agrees = matches!(
                        state_kind_of(input),
                        Some(StateKind::MemoryRead(class)) if class == merge_class
                    );
                    if !agrees {
                        errs.push(RVSDGVerificationError::StateMergeClassMismatch {
                            merge: value,
                            input,
                        });
                    }
                }
            }
        }

        for region_index in 0..self.regions.len() {
            let region = RegionId(region_index as u32);

            // State stays out of the value interface...
            for &param in self.region_params(region) {
                if state_kind_of(param).is_some() {
                    errs.push(RVSDGVerificationError::StateTypedRegionInterface {
                        region,
                        value: param,
                    });
                }
            }
            for &result in self.region_results(region) {
                if state_kind_of(result).is_some() {
                    errs.push(RVSDGVerificationError::StateTypedRegionInterface {
                        region,
                        value: result,
                    });
                }
            }
            // ...and out of the data operands. Projections read data out
            // of state-typed nodes by design; merge inputs are the read
            // evidence, class-checked above.
            for &node in self.region_nodes(region) {
                if matches!(
                    self.get_value_kind(node),
                    ValueKind::Project { .. } | ValueKind::StateMerge { .. }
                ) {
                    continue;
                }
                self.for_each_value_operand(node, |operand| {
                    if state_kind_of(operand).is_some() {
                        errs.push(RVSDGVerificationError::StateTypedDataOperand {
                            user: node,
                            operand,
                        });
                    }
                });

                // A construct's projection run must be data projections
                // first, then at most one memory state projection, then
                // at most one io projection -- the shape consumers of
                // construct_state_projections rely on.
                if self.get_value_kind(node).is_construct() {
                    let mut last_rank = 0u8;
                    let mut id = ValueId(node.0 + 1);
                    while (id.0 as usize) < self.value_kinds.len()
                        && matches!(
                            self.get_value_kind(id),
                            ValueKind::Project { call, .. } if *call == node
                        )
                    {
                        let rank = match state_kind_of(id) {
                            None => 0,
                            Some(StateKind::MemoryRead(_) | StateKind::MemoryWrite(_)) => 1,
                            Some(StateKind::InputOutput) => 2,
                        };
                        let out_of_order = rank < last_rank || (rank == last_rank && rank != 0);
                        if out_of_order {
                            errs.push(RVSDGVerificationError::ConstructStateProjectionsMalformed {
                                construct: node,
                            });
                            break;
                        }
                        last_rank = rank;
                        id = ValueId(id.0 + 1);
                    }
                }
            }

            // Tails: one well-shaped entry per chain, [memory, io].
            for (tail, side) in [
                (self.region_state_params(region), "params"),
                (self.region_state_results(region), "results"),
            ] {
                if tail.len() != 2 {
                    errs.push(RVSDGVerificationError::StateTailWrongArity {
                        region,
                        side,
                        len: tail.len(),
                    });
                    continue;
                }
                if !memory_shaped(tail[0]) {
                    errs.push(RVSDGVerificationError::StateTailWrongChain {
                        region,
                        value: tail[0],
                        chain: "memory",
                    });
                }
                if !io_shaped(tail[1]) {
                    errs.push(RVSDGVerificationError::StateTailWrongChain {
                        region,
                        value: tail[1],
                        chain: "io",
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        AliasClassId, Linkage, RVSDGMod, StateKind, ValueId, ValueKind,
        function_graph::FunctionGraph,
        types::{I32, PtrType, TypeRef},
        verify::RVSDGVerificationError,
    };

    fn ptr_ty(m: &mut RVSDGMod) -> TypeRef {
        TypeRef::Ptr(m.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        }))
    }

    fn nodes_of(graph: &FunctionGraph, want: fn(&ValueKind) -> bool) -> Vec<ValueId> {
        (0..graph.value_kinds.len() as u32)
            .map(ValueId)
            .filter(|&id| want(graph.get_value_kind(id)))
            .collect()
    }

    /// One function with an alloca, two loads (fanning out) and a store
    /// (consuming their merge), plus a live add of the loaded values.
    fn build_memory_module() -> RVSDGMod {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut m);
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let one = rb.const_i32(1);
            let slot = rb.alloca(I32, one, ptr, None);
            rb.store(slot, x, None, false);
            let a = rb.load(slot, I32, None, false);
            let b = rb.load(slot, I32, None, false);
            let sum = rb.binary(
                crate::rvsdg::BinaryOp::Add,
                crate::rvsdg::ArithFlags::default(),
                a,
                b,
                I32,
            );
            rb.store(slot, sum, None, false);
            Ok(vec![sum])
        })
        .unwrap();
        m
    }

    #[test]
    fn producer_with_data_type_is_caught() {
        let mut m = build_memory_module();
        let graph = m.graphs[0].as_mut().unwrap();
        let store = nodes_of(graph, |k| matches!(k, ValueKind::Store { .. }))[0];
        graph.value_types[store.0 as usize] = I32;

        let errs = graph.verify(&m.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::StateProducerTypeWrong { .. })),
            "expected a producer-type error, got: {errs:?}"
        );
    }

    #[test]
    fn state_typed_computation_is_caught() {
        let mut m = build_memory_module();
        let graph = m.graphs[0].as_mut().unwrap();
        let add = nodes_of(graph, |k| matches!(k, ValueKind::Binary { .. }))[0];
        graph.value_types[add.0 as usize] = TypeRef::State(StateKind::MemoryWrite(AliasClassId(0)));

        let errs = graph.verify(&m.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::StateTypedNonProducer(_))),
            "expected a state-typed-non-producer error, got: {errs:?}"
        );
    }

    #[test]
    fn state_flowing_into_data_operand_is_caught() {
        let mut m = build_memory_module();
        let graph = m.graphs[0].as_mut().unwrap();
        let store = nodes_of(graph, |k| matches!(k, ValueKind::Store { .. }))[0];
        let add = nodes_of(graph, |k| matches!(k, ValueKind::Binary { .. }))[0];
        match &mut graph.value_kinds[add.0 as usize] {
            ValueKind::Binary { left, .. } => *left = store,
            other => panic!("expected the add, got {other:?}"),
        }

        let errs = graph.verify(&m.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::StateTypedDataOperand { .. })),
            "expected a state-in-data-operand error, got: {errs:?}"
        );
    }

    #[test]
    fn merge_joining_foreign_class_is_caught() {
        let mut m = build_memory_module();
        let graph = m.graphs[0].as_mut().unwrap();
        let load = nodes_of(graph, |k| matches!(k, ValueKind::Load { .. }))[0];
        assert!(
            !nodes_of(graph, |k| matches!(k, ValueKind::StateMerge { .. })).is_empty(),
            "fixture must produce a merge (two loads before a store)"
        );
        graph.value_types[load.0 as usize] = TypeRef::State(StateKind::MemoryRead(AliasClassId(1)));

        let errs = graph.verify(&m.tables);
        assert!(
            errs.iter()
                .any(|e| matches!(e, RVSDGVerificationError::StateMergeClassMismatch { .. })),
            "expected a merge-class error, got: {errs:?}"
        );
    }

    /// A construct's projection run must be data, then at most one
    /// memory state projection, then at most one io projection.
    /// Retyping the memory projection as io produces a duplicate.
    #[test]
    fn duplicated_construct_state_projection_is_caught() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let flag = rb.constant(I32, crate::rvsdg::ConstValue::Int(1));
            let predicate = rb.match_op(
                flag,
                &[crate::rvsdg::MatchArm {
                    value: 1,
                    alternative: 0,
                }],
                1,
                2,
            );
            let res = rb.gamma(
                predicate,
                &[x],
                |rb| Ok(vec![rb.param(0)]),
                |rb| Ok(vec![rb.param(0)]),
            )?;
            Ok(vec![res.result(0)])
        })
        .unwrap();

        let graph = m.graphs[0].as_mut().unwrap();
        let gamma = nodes_of(graph, |k| matches!(k, ValueKind::Gamma { .. }))[0];
        let memory_projection = graph
            .construct_state_projections(gamma)
            .memory
            .expect("construction creates both state projections");
        graph.value_types[memory_projection.0 as usize] = TypeRef::State(StateKind::InputOutput);

        let errs = graph.verify(&m.tables);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                RVSDGVerificationError::ConstructStateProjectionsMalformed { .. }
            )),
            "expected a malformed-projections error, got: {errs:?}"
        );
    }

    #[test]
    fn io_tail_holding_memory_value_is_caught() {
        let mut m = build_memory_module();
        let graph = m.graphs[0].as_mut().unwrap();
        // Point the body's io state result at a store: a memory-chain
        // value in the io slot.
        let store = nodes_of(graph, |k| matches!(k, ValueKind::Store { .. }))[0];
        let body = graph.regions[0].clone();
        body.state_results_mut(&mut graph.value_pool)[1] = store;

        let errs = graph.verify(&m.tables);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                RVSDGVerificationError::StateTailWrongChain { chain: "io", .. }
            )),
            "expected an io-tail chain error, got: {errs:?}"
        );
    }
}
