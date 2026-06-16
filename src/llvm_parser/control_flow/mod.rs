//! **The CFG -> RVSDG control-flow construction pipeline** (Bahmann, Reissmann,
//! Jahre, Meyer 2015, section 4). Turns an LLVM function's control flow graph
//! into the RVSDG's gamma (branch) and theta (loop) nodes, in two phases:
//!
//! 1. [`restructure`] -- restructure the CFG into a Structured Region Tree
//!    ([`rst`]): the control-structure-only intermediate. Entry point:
//!    [`restructure::restructure_fn`].
//! 2. [`construct`] -- walk the RST and emit the RVSDG. Entry point:
//!    [`RegionLowerer::construct`](crate::llvm_parser::instructions::RegionLowerer).
//!
//! [`analysis`] holds the region/CFG analyses both phases share: continuation
//! points (section 4.2), the irreducible-loop dispatch-dominator table (section
//! 4.1), and phi-driven region live-ins.

pub mod analysis;
pub mod construct;
pub mod restructure;
pub mod rst;

#[cfg(test)]
use crate::llvm_parser::{block_mapper::BasicBlockId, instructions::RegionLowerer};

/// Build the per-function [`FnCtx`](crate::llvm_parser::FnCtx) (block interning,
/// dominators, strongly-connected-component tree, dispatch tables) and run `f`
/// with it. Test-only harness for the construction tests.
#[cfg(test)]
fn with_test_fn_ctx<R>(
    func: &llvm_ir::Function,
    module: &llvm_ir::Module,
    f: impl FnOnce(&crate::llvm_parser::FnCtx) -> R,
) -> R {
    use crate::llvm_parser::{
        dominance::{ForwardView, ReverseView, compute_dominance},
        intern_blocks_and_arcs,
        scc::SccTree,
    };

    let bb_mapper = intern_blocks_and_arcs(func);
    let exit_block_id = *bb_mapper.get_exit_expect();
    let immediate_dominators = compute_dominance(&ForwardView {
        nodes: &bb_mapper.blocks,
        entry: BasicBlockId(0),
    });
    let post_immediate_dominators = compute_dominance(&ReverseView {
        nodes: &bb_mapper.blocks,
        exit: exit_block_id,
    });
    let scc_tree = SccTree::build(&bb_mapper);
    let scc_entry_block_to_id = scc_tree.entry_block_to_node(bb_mapper.blocks.len());
    let multi_entry_dispatch =
        crate::llvm_parser::control_flow::analysis::loops::compute_multi_entry_dispatch(
            &scc_tree,
            &immediate_dominators,
            bb_mapper.blocks.len(),
        );
    let fn_ctx = crate::llvm_parser::FnCtx {
        llvm_mod: module,
        func,
        bb_mapper: &bb_mapper,
        scc_tree: &scc_tree,
        scc_entry_block_to_id: &scc_entry_block_to_id,
        multi_entry_dispatch: &multi_entry_dispatch,
        immediate_dominators: &immediate_dominators,
        post_immediate_dominators: &post_immediate_dominators,
        exit_block_id,
    };
    f(&fn_ctx)
}

/// Lower `func`'s body into `rvsdg` with the two-phase restructure + construct
/// driver (`restructure::restructure_fn` then `RegionLowerer::construct`).
#[cfg(test)]
fn lower_body_with_rst_driver(
    rvsdg: &mut crate::rvsdg::RVSDGMod,
    func: &llvm_ir::Function,
    module: &llvm_ir::Module,
) -> color_eyre::Result<()> {
    use crate::{
        llvm_parser::control_flow::{construct::ConstructExit, restructure::restructure_fn},
        rvsdg::func::FnResult,
    };

    let rvsdg_fn_id = rvsdg.get_func_by_name(&func.name).unwrap().id;
    with_test_fn_ctx(func, module, |fn_ctx| {
        let rst = restructure_fn(fn_ctx)?;
        rvsdg.define_fn(rvsdg_fn_id, |rb, state| {
            let mut builder = RegionLowerer::new(rb, fn_ctx);
            for (i, param) in func.parameters.iter().enumerate() {
                let value = builder.rb.param(i as u32);
                builder.name_to_value.insert(param.name.clone(), value);
            }
            let exit = builder.construct(&rst, state, None, &[])?;
            let (state, values) = match exit {
                ConstructExit::Returned { state, values } => (state, values),
                ConstructExit::AtBoundary { state, .. } => (state, vec![]),
                ConstructExit::Diverge { state } => (state, vec![]),
            };
            Ok(FnResult { state, values })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{RVSDGMod, func::FnDecl};
    use inkwell::{OptimizationLevel, context::Context};
    use llvm_ir::Module;
    use std::sync::Mutex;

    // llvm-ir 0.11.3 lazily inits a global attribute table on first parse that
    // races under concurrent test threads; serialise parses.
    static LLVM_PARSE_LOCK: Mutex<()> = Mutex::new(());

    /// Parse `ir` and build every function's body with the two-phase
    /// restructure + construct driver, returning the RVSDG module (no JIT).
    fn build_rst_module(ir: &str) -> RVSDGMod {
        let module = {
            let _guard = LLVM_PARSE_LOCK.lock().unwrap();
            Module::from_ir_str(ir).expect("parse test IR")
        };
        let mut rvsdg = RVSDGMod::new_host("test".to_string());
        for func in &module.functions {
            let decl = FnDecl::from_fn(func, &mut rvsdg.types, &module).expect("fn decl");
            rvsdg.declare_fn_full(decl);
        }
        for func in &module.functions {
            lower_body_with_rst_driver(&mut rvsdg, func, &module).expect("rst driver lowering");
        }
        rvsdg
    }

    /// Parse `ir`, build with the two-phase driver, then JIT and call the
    /// no-argument `i32`-returning function `fn_name`.
    fn build_and_jit_i32_rst(ir: &str, fn_name: &str) -> i32 {
        let rvsdg = build_rst_module(ir);

        crate::init_llvm_native().expect("init native target");
        let context = Context::create();
        let llvm_module = rvsdg.lower_to_llvm_module(&context).expect("lower to llvm");
        let engine = llvm_module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("jit engine");
        let func = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> i32>(fn_name)
                .expect("jit function")
        };
        unsafe { func.call() }
    }

    #[test]
    fn rst_straight_line() {
        let ir = r#"
define i32 @f() {
entry:
  %a = add i32 40, 2
  %b = mul i32 %a, 1
  ret i32 %b
}
"#;
        assert_eq!(42, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_if_else() {
        let ir = r#"
define i32 @f() {
entry:
  %c = icmp sgt i32 5, 3
  br i1 %c, label %t, label %e
t:
  br label %m
e:
  br label %m
m:
  %r = phi i32 [ 10, %t ], [ 20, %e ]
  ret i32 %r
}
"#;
        assert_eq!(10, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_if_else_live_in() {
        let ir = r#"
define i32 @f() {
entry:
  %base = add i32 5, 0
  %c = icmp sgt i32 %base, 3
  br i1 %c, label %t, label %e
t:
  %tv = add i32 %base, 100
  br label %m
e:
  %ev = add i32 %base, 200
  br label %m
m:
  %r = phi i32 [ %tv, %t ], [ %ev, %e ]
  ret i32 %r
}
"#;
        assert_eq!(105, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_switch_single_join() {
        let ir = r#"
define i32 @f() {
entry:
  switch i32 1, label %d [
    i32 0, label %c0
    i32 1, label %c1
  ]
c0:
  br label %m
c1:
  br label %m
d:
  br label %m
m:
  %r = phi i32 [ 100, %c0 ], [ 200, %c1 ], [ 300, %d ]
  ret i32 %r
}
"#;
        assert_eq!(200, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_recursive_pdemux() {
        let ir = r#"
define i32 @f(i32 %a, i32 %b, i32 %x) {
entry:
  %c0 = icmp ne i32 %a, 0
  br i1 %c0, label %sw, label %contB
sw:
  switch i32 %b, label %toC [ i32 7, label %toB ]
toB:
  br label %contB
toC:
  br label %contC
contB:
  %vb = mul i32 %x, 3
  br label %contC
contC:
  %r = phi i32 [ %vb, %contB ], [ %x, %toC ]
  ret i32 %r
}
define i32 @main() {
  %p = call i32 @f(i32 0, i32 0, i32 5)
  %q = call i32 @f(i32 1, i32 7, i32 5)
  %s = call i32 @f(i32 1, i32 9, i32 5)
  %t1 = add i32 %p, %q
  %t2 = add i32 %t1, %s
  ret i32 %t2
}
"#;
        let rvsdg = build_rst_module(ir);
        let muls = count_values(&rvsdg, |k| {
            matches!(
                k,
                crate::rvsdg::ValueKind::Binary {
                    op: crate::rvsdg::BinaryOp::Mul,
                    ..
                }
            )
        });
        assert_eq!(
            1, muls,
            "recursive p-demux re-lowered the shared continuation"
        );
        assert_eq!(35, build_and_jit_i32_rst(ir, "main"));
    }

    #[test]
    fn rst_recursive_pdemux_three_level() {
        let ir = r#"
define i32 @f3(i32 %a, i32 %b, i32 %c, i32 %x) {
entry:
  %ca = icmp ne i32 %a, 0
  br i1 %ca, label %r1, label %contB
r1:
  %cb = icmp ne i32 %b, 0
  br i1 %cb, label %r2, label %toB1
r2:
  %cc = icmp ne i32 %c, 0
  br i1 %cc, label %toC, label %toB2
toB1:
  br label %contB
toB2:
  br label %contB
toC:
  br label %contC
contB:
  %vb = mul i32 %x, 7
  br label %contC
contC:
  %r = phi i32 [ %vb, %contB ], [ %x, %toC ]
  ret i32 %r
}
define i32 @main() {
  %p = call i32 @f3(i32 0, i32 0, i32 0, i32 2)
  %q = call i32 @f3(i32 1, i32 1, i32 1, i32 2)
  %s = call i32 @f3(i32 1, i32 1, i32 0, i32 2)
  %t1 = add i32 %p, %q
  %t2 = add i32 %t1, %s
  ret i32 %t2
}
"#;
        let rvsdg = build_rst_module(ir);
        let muls = count_values(&rvsdg, |k| {
            matches!(
                k,
                crate::rvsdg::ValueKind::Binary {
                    op: crate::rvsdg::BinaryOp::Mul,
                    ..
                }
            )
        });
        assert_eq!(
            1, muls,
            "3-level recursive p-demux re-lowered the shared continuation"
        );
        assert_eq!(30, build_and_jit_i32_rst(ir, "main"));
    }

    #[test]
    fn rst_do_while_counter() {
        let ir = r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %header ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %header ]
  %acc.next = add i32 %acc, %i
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, 5
  br i1 %c, label %header, label %exit
exit:
  %r = phi i32 [ %acc.next, %header ]
  ret i32 %r
}
"#;
        assert_eq!(10, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_test_first_while() {
        let ir = r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %body ]
  %c = icmp slt i32 %i, 5
  br i1 %c, label %body, label %exit
body:
  %acc.next = add i32 %acc, %i
  %i.next = add i32 %i, 1
  br label %header
exit:
  %r = phi i32 [ %acc, %header ]
  ret i32 %r
}
"#;
        assert_eq!(10, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_if_in_loop_clone_free() {
        let ir = r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc2, %latch ]
  %odd = and i32 %i, 1
  %isodd = icmp eq i32 %odd, 1
  br i1 %isodd, label %add, label %skip
add:
  %acc.a = add i32 %acc, %i
  br label %latch
skip:
  br label %latch
latch:
  %acc2 = phi i32 [ %acc.a, %add ], [ %acc, %skip ]
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, 6
  br i1 %c, label %header, label %exit
exit:
  %r = phi i32 [ %acc2, %latch ]
  ret i32 %r
}
"#;
        let rvsdg = build_rst_module(ir);
        let icmps = count_values(&rvsdg, |k| {
            matches!(k, crate::rvsdg::ValueKind::ICmp { .. })
        });
        assert_eq!(2, icmps, "loop-body branch cloned the shared latch");
        assert_eq!(9, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_multi_exit_loop() {
        let ir = r#"
define i32 @f() {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %big = icmp sgt i32 %i, 3
  br i1 %big, label %exitA, label %latch
latch:
  %i.next = add i32 %i, 1
  %done = icmp sge i32 %i.next, 10
  br i1 %done, label %exitB, label %header
exitA:
  br label %join
exitB:
  br label %join
join:
  %r = phi i32 [ 100, %exitA ], [ 200, %exitB ]
  ret i32 %r
}
"#;
        assert_eq!(100, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_irreducible_clone_free() {
        let ir = r#"
define dso_local i32 @f(i32 %0, i32 %1) {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %5
4:
  br label %8
5:
  br label %6
6:
  %.08 = phi i32 [ %16, %18 ], [ %1, %5 ]
  %.0 = phi i32 [ %.2, %18 ], [ 0, %5 ]
  %7 = add nsw i32 %.0, 10
  br label %8
8:
  %.19 = phi i32 [ %1, %4 ], [ %.08, %6 ]
  %.1 = phi i32 [ 0, %4 ], [ %7, %6 ]
  %9 = and i32 %.1, 1
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %13
11:
  %12 = add nsw i32 %.1, 100
  br label %15
13:
  %14 = add nsw i32 %.1, 1
  br label %15
15:
  %.2 = phi i32 [ %12, %11 ], [ %14, %13 ]
  %16 = sub nsw i32 %.19, 1
  %17 = icmp sgt i32 %16, 0
  br i1 %17, label %18, label %19
18:
  br label %6
19:
  ret i32 %.2
}
define i32 @main() {
  %r = call i32 @f(i32 0, i32 4)
  ret i32 %r
}
"#;
        let rvsdg = build_rst_module(ir);
        let icmps = count_values(&rvsdg, |k| {
            matches!(k, crate::rvsdg::ValueKind::ICmp { .. })
        });
        assert_eq!(3, icmps, "irreducible loop body cloned a shared block");
        assert_eq!(331, build_and_jit_i32_rst(ir, "main"));
    }

    #[test]
    fn rst_nonreconverging_return() {
        let ir = r#"
define i32 @f() {
entry:
  %c = icmp sgt i32 5, 3
  br i1 %c, label %t, label %e
t:
  ret i32 11
e:
  ret i32 22
}
"#;
        assert_eq!(11, build_and_jit_i32_rst(ir, "f"));
    }

    #[test]
    fn rst_nested_loops() {
        let ir = r#"
define i32 @f() {
entry:
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.outer, %latch ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %acc.in = phi i32 [ %acc, %outer ], [ %acc.next, %inner ]
  %acc.next = add i32 %acc.in, 1
  %j.next = add i32 %j, 1
  %jc = icmp slt i32 %j.next, 3
  br i1 %jc, label %inner, label %latch
latch:
  %acc.outer = phi i32 [ %acc.next, %inner ]
  %i.next = add i32 %i, 1
  %ic = icmp slt i32 %i.next, 4
  br i1 %ic, label %outer, label %exit
exit:
  %r = phi i32 [ %acc.outer, %latch ]
  ret i32 %r
}
"#;
        assert_eq!(12, build_and_jit_i32_rst(ir, "f"));
    }

    /// Count RVSDG nodes of a given `ValueKind` discriminant by matcher.
    fn count_values(rvsdg: &RVSDGMod, pred: impl Fn(&crate::rvsdg::ValueKind) -> bool) -> usize {
        rvsdg.values.iter().filter(|v| pred(&v.kind)).count()
    }
}
