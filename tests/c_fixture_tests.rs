use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

use lang_rvsdg::{Cli, run_cli};
use tempfile::TempDir;

fn test_c_file(input: &str) {
    let rvsdg_res = run_example_rvsdg(input);
    let clang_res = run_example_clang(input);
    assert_eq!(rvsdg_res as i32, clang_res);
}

fn run_example_rvsdg(input: &str) -> u8 {
    let cli = Cli::get_run_integration(input.to_string());
    run_cli(&cli).unwrap().unwrap()
}

fn run_example_clang(input: &str) -> i32 {
    let tmp_dir = TempDir::new().expect("failed to create temp dir").keep();
    let path_str = {
        let mut path = PathBuf::new();
        path.push(tmp_dir);
        path.push("out");
        path.to_str().expect("failed to construct path").to_string()
    };
    let clang_status = Command::new("clang")
        .args([input, "-O2", "-o", &path_str])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .status()
        .expect("failed to start clang");
    if !clang_status.success() {
        panic!("failed to compile with clang");
    }

    let bin_status = Command::new(&path_str)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .status()
        .expect("failed to start clang");
    bin_status.code().unwrap()
}

#[test]
fn test_00_straight_line() {
    test_c_file("tests/fixtures/c/00_straight_line.c");
}

#[test]
fn test_01_call_chain() {
    test_c_file("tests/fixtures/c/01_call_chain.c");
}

#[test]
fn test_02_if_else() {
    test_c_file("tests/fixtures/c/02_if_else.c");
}

#[test]
fn test_03_do_while() {
    test_c_file("tests/fixtures/c/03_do_while.c");
}

#[test]
fn test_04_while_loop() {
    test_c_file("tests/fixtures/c/04_while_loop.c");
}

#[test]
fn test_05_loop_with_break() {
    test_c_file("tests/fixtures/c/05_loop_with_break.c");
}

#[test]
fn test_06_nested_loops() {
    test_c_file("tests/fixtures/c/06_nested_loops.c");
}

#[test]
fn test_07_switch() {
    test_c_file("tests/fixtures/c/07_switch.c");
}

// Irreducible control flow (multi-entry loop): the §4.1 `q` entry-predicate,
// built fully in-tree — the loop is lowered once at the entries' dispatch
// dominator, with q computed in the entry region and a single q-dispatch theta.
#[test]
fn test_08_irreducible() {
    test_c_file("tests/fixtures/c/08_irreducible.c");
}

#[test]
fn test_09_triple_nested() {
    test_c_file("tests/fixtures/c/09_triple_nested.c");
}

#[test]
fn test_10_loop_in_gamma() {
    test_c_file("tests/fixtures/c/10_loop_in_gamma.c");
}

#[test]
fn test_11_printf() {
    test_c_file("tests/fixtures/c/11_printf.c");
}

#[test]
fn test_12_zero_iter() {
    test_c_file("tests/fixtures/c/12_zero_iter.c");
}

#[test]
fn test_13_test_first_no_phis() {
    test_c_file("tests/fixtures/c/13_test_first_no_phis.c");
}

#[test]
fn test_14_loop_internal_join() {
    test_c_file("tests/fixtures/c/14_loop_internal_join.c");
}

#[test]
fn test_basic() {
    test_c_file("tests/fixtures/c/basic.c");
}

// ---------------------------------------------------------------------------
// csmith-blocking gaps. These isolate features csmith emits pervasively but
// the pipeline does not yet handle. Re-enable each as the feature lands.
// ---------------------------------------------------------------------------

// Short-circuit `&&`: empty pass-through arm whose join-phi value flows
// along the head->join edge. Handled (Bahmann §4.2 empty-branch rule).
#[test]
fn test_15_short_circuit_and() {
    test_c_file("tests/fixtures/c/15_short_circuit_and.c");
}

// Short-circuit `||`: same shape as 15.
#[test]
fn test_16_short_circuit_or() {
    test_c_file("tests/fixtures/c/16_short_circuit_or.c");
}

// `break` before body work: loop-closed phi bound at a non-natural exit.
// Handled — captured through an extra theta loop_var slot (the header
// dest's pre-update value).
#[test]
fn test_17_break_before_body() {
    test_c_file("tests/fixtures/c/17_break_before_body.c");
}

// ---------------------------------------------------------------------------
// Remaining construction gaps (see gap.md). Each fixture below fails today;
// uncomment as the corresponding stage lands.
// ---------------------------------------------------------------------------

// §4.2: switch fall-through (shared continuation tail). Handled via
// path-aware interior-phi resolution + exit-predecessor arm contribution.
#[test]
fn test_18_switch_fallthrough() {
    test_c_file("tests/fixtures/c/18_switch_fallthrough.c");
}

// §4.2: branch arms reaching a shared continuation (cross edges).
#[test]
fn test_19_branch_cross_edges() {
    test_c_file("tests/fixtures/c/19_branch_cross_edges.c");
}

// §4.1 body walker: switch inside a loop body. Handled via lower_body_switch
// (n-arm gamma in the body, sharing the switch selector with the acyclic path).
#[test]
fn test_20_switch_in_loop() {
    test_c_file("tests/fixtures/c/20_switch_in_loop.c");
}

// §4.1: break out of a nested loop. The inner loop's exit lands at an
// in-body switch (continue-vs-break the outer loop); lowered as an n-arm
// gamma on the i32 selector now that the backend picks branch-vs-switch by
// condition type.
#[test]
fn test_21_break_nested_loop() {
    test_c_file("tests/fixtures/c/21_break_nested_loop.c");
}

// §4.1 body walker: early return inside a loop body.
// Unblocked by the post-dominance fix (exit-unreachable blocks now get None).
#[test]
fn test_22_return_in_loop() {
    test_c_file("tests/fixtures/c/22_return_in_loop.c");
}

// §4.2 rest: an `__builtin_unreachable()` arm. Unblocked by the
// post-dominance fix.
#[test]
fn test_24_unreachable_arm() {
    test_c_file("tests/fixtures/c/24_unreachable_arm.c");
}

// §4.2 rest: noreturn-call arm. The void-returning `abort()` call now
// lowers (indirect-call fn type built via void_type()).
#[test]
fn test_23_noreturn_arm() {
    test_c_file("tests/fixtures/c/23_noreturn_arm.c");
}

// §4.2 rest / §4.1: infinite-loop arm. The zero-exit loop now lowers to a
// theta whose continuation is the (dead) function exit.
#[test]
fn test_25_infinite_loop_arm() {
    test_c_file("tests/fixtures/c/25_infinite_loop_arm.c");
}

// ---------------------------------------------------------------------------
// Non-reconverging / multi-exit control flow (continuation-predicate area).
// These all PASS: loop-simplify generally unifies a loop's exits into one
// exit block, so the common multi-return / break+return / goto-multi-exit /
// nested shapes are handled. They lock in that robustness.
// ---------------------------------------------------------------------------

#[test]
fn test_26_loop_two_returns() {
    test_c_file("tests/fixtures/c/26_loop_two_returns.c");
}

#[test]
fn test_27_loop_break_and_return() {
    test_c_file("tests/fixtures/c/27_loop_break_and_return.c");
}

#[test]
fn test_28_switch_in_loop_mixed() {
    test_c_file("tests/fixtures/c/28_switch_in_loop_mixed.c");
}

#[test]
fn test_29_loop_chain() {
    test_c_file("tests/fixtures/c/29_loop_chain.c");
}

#[test]
fn test_30_break_outer_flag() {
    test_c_file("tests/fixtures/c/30_break_outer_flag.c");
}

#[test]
fn test_31_goto_multi_exit() {
    test_c_file("tests/fixtures/c/31_goto_multi_exit.c");
}

#[test]
fn test_32_do_while_return() {
    test_c_file("tests/fixtures/c/32_do_while_return.c");
}

#[test]
fn test_33_loop_exit_into_loop() {
    test_c_file("tests/fixtures/c/33_loop_exit_into_loop.c");
}

#[test]
fn test_34_loop_exit_into_infinite() {
    test_c_file("tests/fixtures/c/34_loop_exit_into_infinite.c");
}

// Multi-exit loop with one `unreachable` exit target (minimal reduction of a
// csmith finding). Handled: exit-unreachable targets are excluded from the
// join and get a poison dispatch arm.
#[test]
fn test_35_loop_unreachable_exit() {
    test_c_file("tests/fixtures/c/35_loop_unreachable_exit.c");
}

// Tier 36 — global struct with a constant `struct` initializer. The initializer
// is a constant aggregate (lowered with const_struct); the fields are read via
// typed constant GEPs `getelementptr(%struct.S, @g, 0, k)`, including a nested
// `getelementptr([N x i32], <inner-gep>, 0, k)` for the array field. The
// constant-GEP source type is recovered from the index shape (1 index → i8 byte
// form, many → typed access over the base's pointee type).
#[test]
fn test_36_global_struct() {
    test_c_file("tests/fixtures/c/36_global_struct.c");
}

// Regression for seed 7000: an unreachable block whose edge points into a loop
// fabricates a phantom second entry vertex (spurious irreducibility). Handled by
// excluding arcs out of unreachable blocks when building the CFG.
#[test]
fn test_37_dead_goto_into_loop() {
    test_c_file("tests/fixtures/c/37_dead_goto_into_loop.c");
}
