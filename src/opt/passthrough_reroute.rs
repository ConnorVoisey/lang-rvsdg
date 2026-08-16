//! Passthrough reroute: state chains step around constructs they merely
//! pass through.
//!
//! Construction threads every chain through every construct, so a
//! construct is on a chain even when no subregion touches it -- its
//! state projection is the chain's current afterwards, and the exit
//! tail roots it. Liveness alone can therefore never collect a pure
//! construct. This pass restores that: for each construct and each
//! chain slot, if EVERY subregion passes the slot through (exit tail
//! entry == entry tail entry), every consumer of the construct's state
//! projection is redirected to the construct's own chain input (the
//! entry tail entry, a parent-side value). A construct bypassed on all
//! chains and dead in its data outputs is then unreachable from any
//! root, and the following dead-node-elimination removes it whole.
//!
//! Constructs are visited in ascending id order, which is
//! innermost-first (subregions are emitted before their construct is
//! assembled): bypassing an inner construct is what makes the enclosing
//! subregion's exit equal its entry, so the enclosing construct's
//! passthrough becomes visible in the same sweep. Redirections resolve
//! transitively through the map for the same reason.

use rustc_hash::FxHashMap;

use crate::rvsdg::{RVSDGMod, ValueId, function_graph::FunctionGraph};

/// Counters this pass can produce cheaply; whole-graph deltas are the
/// driver's job (see [`super::PassReport`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct PassthroughEffects {
    /// Memory chain slots rerouted around a construct.
    pub memory_slots_rerouted: u64,
    /// Io chain slots rerouted around a construct.
    pub io_slots_rerouted: u64,
}

impl RVSDGMod {
    pub fn opt_passthrough_reroute(&mut self) -> color_eyre::Result<PassthroughEffects> {
        let mut effects = PassthroughEffects::default();
        for graph in self.graphs.iter_mut().flatten() {
            graph.reroute_passthrough_state(&mut effects);
        }
        Ok(effects)
    }
}

/// Follow the redirection map to its end: an entry's target may itself
/// have been bypassed (an inner construct's projection).
fn resolve(map: &FxHashMap<ValueId, ValueId>, mut value: ValueId) -> ValueId {
    while let Some(&next) = map.get(&value) {
        value = next;
    }
    value
}

impl FunctionGraph {
    fn reroute_passthrough_state(&mut self, effects: &mut PassthroughEffects) {
        // projection-of-bypassed-slot -> the construct's chain input.
        let mut map: FxHashMap<ValueId, ValueId> = FxHashMap::default();

        for index in 0..self.value_kinds.len() {
            let construct = ValueId(index as u32);
            let subregions = self.construct_subregions(construct);
            if subregions.is_empty() {
                continue;
            }

            // A slot is passthrough when every subregion's exit entry
            // resolves to its entry entry. Entry tails are identical
            // across subregions (verified), so slot inputs come off the
            // first. Projections are found by type: a previous run may
            // already have removed a bypassed slot's projection.
            debug_assert_eq!(self.region_state_params(subregions[0]).len(), 2);
            let projections = self.construct_state_projections(construct);
            for (slot, projection) in [(0usize, projections.memory), (1, projections.io)] {
                let passthrough = subregions.iter().all(|&sub| {
                    let entry = self.region_state_params(sub)[slot];
                    let exit = self.region_state_results(sub)[slot];
                    resolve(&map, exit) == resolve(&map, entry)
                });
                let (true, Some(projection)) = (passthrough, projection) else {
                    continue;
                };
                let input = resolve(&map, self.region_state_params(subregions[0])[slot]);
                map.insert(projection, input);
                match slot {
                    0 => effects.memory_slots_rerouted += 1,
                    _ => effects.io_slots_rerouted += 1,
                }
            }
        }

        if map.is_empty() {
            return;
        }

        // Apply the redirections everywhere state is consumed: the state
        // operands of ops, and every region's state tails (a subregion's
        // ENTRY tail can hold a bypassed sibling's projection; results
        // tails close chains that may now end earlier). Merge inputs are
        // read evidence -- always loads, never projections -- and data
        // operands cannot reference state values (verified), so neither
        // needs a walk.
        for kind in self.value_kinds.iter_mut() {
            kind.for_each_state_operand_mut(|state| state.0 = resolve(&map, state.0));
        }
        for region_index in 0..self.regions.len() {
            let region = self.regions[region_index].clone();
            for entry in region.state_params_mut(&mut self.value_pool) {
                *entry = resolve(&map, *entry);
            }
            for entry in region.state_results_mut(&mut self.value_pool) {
                *entry = resolve(&map, *entry);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rvsdg::{
        ArithFlags, BinaryOp, ConstValue, Linkage, RVSDGMod, RegionId, ValueId, ValueKind,
        function_graph::FunctionGraph,
        types::{BOOL, I32, PtrType, TypeRef},
    };

    fn graph(m: &RVSDGMod) -> &FunctionGraph {
        m.graphs[0].as_ref().unwrap()
    }

    fn count_nodes(m: &RVSDGMod, want: fn(&ValueKind) -> bool) -> usize {
        let g = graph(m);
        (0..g.value_kinds.len() as u32)
            .map(ValueId)
            .filter(|&id| want(g.get_value_kind(id)))
            .count()
    }

    fn ptr_ty(m: &mut RVSDGMod) -> TypeRef {
        TypeRef::Ptr(m.tables.types.intern_ptr(PtrType {
            pointee: Some(I32),
            alias_set: None,
            no_escape: false,
        }))
    }

    /// The case liveness alone cannot handle: a gamma whose data output
    /// is unused and whose arms touch no chain. Reroute steps both
    /// chains around it; elimination then collects the construct, its
    /// arms' contents, and its projections.
    #[test]
    fn dead_pure_gamma_is_removed() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let y = rb.param(1);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let _unused = rb.gamma(
                predicate,
                &[x, y],
                |rb| {
                    let (a, b) = (rb.param(0), rb.param(1));
                    Ok(vec![rb.binary(
                        BinaryOp::Mul,
                        ArithFlags::default(),
                        a,
                        b,
                        I32,
                    )])
                },
                |rb| {
                    let (a, b) = (rb.param(0), rb.param(1));
                    Ok(vec![rb.binary(
                        BinaryOp::Mul,
                        ArithFlags::default(),
                        b,
                        a,
                        I32,
                    )])
                },
            )?;
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            Ok(vec![live])
        })
        .unwrap();
        // Both arm regions are pure: every state slot passes through.
        {
            let g = graph(&m);
            let arms: Vec<_> = g
                .regions
                .iter()
                .filter(|r| r.owner != ValueId::INVALID)
                .collect();
            assert_eq!(arms.len(), 2);
            assert!(arms.iter().all(|r| r.is_pure(g)));
        }

        m.optimise_default(true).unwrap();

        assert_eq!(count_nodes(&m, |k| matches!(k, ValueKind::Gamma { .. })), 0);
        assert_eq!(
            count_nodes(
                &m,
                |k| matches!(k, ValueKind::Binary { op, .. } if *op == BinaryOp::Mul)
            ),
            0,
            "arm contents die with the construct"
        );
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Project { .. })),
            0,
            "no projection survives a removed construct"
        );
        assert!(m.verify().is_empty());
    }

    /// A pure gamma whose data output IS used: the construct stays, but
    /// the chains step around it -- the function's exit tails read the
    /// values from before the construct.
    #[test]
    fn live_pure_gamma_is_bypassed_only() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut m);
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let one = rb.const_i32(1);
            let slot = rb.alloca(I32, one, ptr, None);
            rb.store(slot, x, None, false);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let res = rb.gamma(
                predicate,
                &[x],
                |rb| Ok(vec![rb.param(0)]),
                |rb| {
                    let a = rb.param(0);
                    Ok(vec![rb.binary(
                        BinaryOp::Mul,
                        ArithFlags::default(),
                        a,
                        a,
                        I32,
                    )])
                },
            )?;
            Ok(vec![res.result(0)])
        })
        .unwrap();

        let effects = m.opt_passthrough_reroute().unwrap();
        assert_eq!(effects.memory_slots_rerouted, 1);
        assert_eq!(effects.io_slots_rerouted, 1);

        let g = graph(&m);
        let store = (0..g.value_kinds.len() as u32)
            .map(ValueId)
            .find(|&id| matches!(g.get_value_kind(id), ValueKind::Store { .. }))
            .unwrap();
        let exits = g.region_state_results(RegionId(0));
        let entries = g.region_state_params(RegionId(0));
        assert_eq!(
            exits[0], store,
            "the memory chain ends at the store, bypassing the gamma"
        );
        assert_eq!(
            exits[1], entries[1],
            "the io chain passes the whole function through"
        );
        assert!(m.verify().is_empty());

        // The bypassed projections are dead weight; elimination
        // reclaims both while the construct itself stays.
        m.opt_dead_node_elimination().unwrap();
        assert_eq!(count_nodes(&m, |k| matches!(k, ValueKind::Gamma { .. })), 1);
        let g = graph(&m);
        let state_projections = (0..g.value_kinds.len() as u32)
            .map(ValueId)
            .filter(|&id| {
                matches!(g.get_value_kind(id), ValueKind::Project { .. })
                    && matches!(g.get_value_type(id), crate::rvsdg::types::TypeRef::State(_))
            })
            .count();
        assert_eq!(state_projections, 0, "bypassed state projections reclaimed");
        assert!(m.verify().is_empty());
    }

    /// One arm stores: the memory chain must keep flowing through the
    /// construct, while the untouched io chain is still bypassed.
    #[test]
    fn impure_arm_keeps_memory_chain_through_construct() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let ptr = ptr_ty(&mut m);
        let f = m.declare_fn(String::from("f"), &[I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let one = rb.const_i32(1);
            let slot = rb.alloca(I32, one, ptr, None);
            let flag = rb.constant(BOOL, ConstValue::Int(1));
            let predicate = rb.bool_predicate(flag);
            let res = rb.gamma(
                predicate,
                &[x, slot],
                |rb| {
                    let (value, addr) = (rb.param(0), rb.param(1));
                    rb.store(addr, value, None, false);
                    Ok(vec![rb.param(0)])
                },
                |rb| Ok(vec![rb.param(0)]),
            )?;
            Ok(vec![res.result(0)])
        })
        .unwrap();
        // The storing arm is impure, its sibling pure.
        {
            let g = graph(&m);
            let purity: Vec<bool> = g
                .regions
                .iter()
                .filter(|r| r.owner != ValueId::INVALID)
                .map(|r| r.is_pure(g))
                .collect();
            assert_eq!(purity, vec![false, true]);
        }

        let effects = m.opt_passthrough_reroute().unwrap();
        assert_eq!(effects.memory_slots_rerouted, 0);
        assert_eq!(effects.io_slots_rerouted, 1);
        assert!(m.verify().is_empty());

        // The gamma must survive elimination: the exit chain flows
        // through its memory state projection. The rerouted io
        // projection is reclaimed; the memory one stays.
        m.opt_dead_node_elimination().unwrap();
        assert_eq!(count_nodes(&m, |k| matches!(k, ValueKind::Gamma { .. })), 1);
        assert_eq!(
            count_nodes(&m, |k| matches!(k, ValueKind::Store { .. })),
            1,
            "the arm's store survives with its construct"
        );
        let g = graph(&m);
        let gamma = (0..g.value_kinds.len() as u32)
            .map(ValueId)
            .find(|&id| matches!(g.get_value_kind(id), ValueKind::Gamma { .. }))
            .unwrap();
        let projections = g.construct_state_projections(gamma);
        assert!(
            projections.memory.is_some(),
            "the live chain keeps its projection"
        );
        assert!(
            projections.io.is_none(),
            "the bypassed chain's projection is reclaimed"
        );
        assert!(m.verify().is_empty());
    }

    /// Nesting cascades in one sweep: a pure gamma inside a theta whose
    /// slots are all dead. Bypassing the gamma makes the body pure,
    /// which bypasses the theta; elimination removes both.
    #[test]
    fn nested_pure_constructs_cascade() {
        let mut m = RVSDGMod::new_host(String::from("test"));
        let f = m.declare_fn(String::from("f"), &[I32, I32], &[I32], Linkage::Internal);
        m.define_fn(f, |rb| {
            let x = rb.param(0);
            let y = rb.param(1);
            let _unused = rb.theta(&[x, y], |rb| {
                let i = rb.param(0);
                let limit = rb.param(1);
                let flag = rb.constant(BOOL, ConstValue::Int(1));
                let predicate = rb.bool_predicate(flag);
                let res = rb.gamma(
                    predicate,
                    &[i],
                    |rb| Ok(vec![rb.param(0)]),
                    |rb| {
                        let a = rb.param(0);
                        Ok(vec![rb.binary(
                            BinaryOp::Mul,
                            ArithFlags::default(),
                            a,
                            a,
                            I32,
                        )])
                    },
                )?;
                let next = res.result(0);
                let repeat = rb.icmp(crate::rvsdg::ICmpPred::SignedLt, next, limit);
                let condition = rb.bool_predicate(repeat);
                Ok(crate::rvsdg::builder::LoopResult {
                    condition,
                    next_vars: vec![next, limit],
                })
            })?;
            let live = rb.binary(BinaryOp::Add, ArithFlags::default(), x, y, I32);
            Ok(vec![live])
        })
        .unwrap();

        m.optimise_default(true).unwrap();

        assert_eq!(count_nodes(&m, |k| matches!(k, ValueKind::Gamma { .. })), 0);
        assert_eq!(count_nodes(&m, |k| matches!(k, ValueKind::Theta { .. })), 0);
        assert!(m.verify().is_empty());
    }
}
