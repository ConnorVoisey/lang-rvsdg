use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

use lang_rvsdg::{Cli, OutputIntegration, run_cli};
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

/// Two-translation-unit ABI differential: `our_file` goes through the
/// RVSDG pipeline, `clang_file` is compiled by clang, and the two are
/// linked into one binary; the reference build compiles both with clang.
/// An ABI mismatch (byval/sret/extension attributes or calling convention
/// dropped on either side) shows up as a differing exit code.
fn test_c_pair(our_file: &str, clang_file: &str) {
    let tmp_dir = TempDir::new().expect("failed to create temp dir").keep();
    let path_of = |name: &str| {
        tmp_dir
            .join(name)
            .to_str()
            .expect("failed to construct path")
            .to_string()
    };

    // The clang-owned half, as an object file our link step consumes.
    let helper_obj = path_of("helper.o");
    let status = Command::new("clang")
        .args(["-O1", "-w", "-c", clang_file, "-o", &helper_obj])
        .status()
        .expect("failed to start clang");
    assert!(status.success(), "clang failed to compile {clang_file}");

    // Our half, linked against the clang object.
    let ours_bin = path_of("ours");
    let cli = Cli::get_output_integration(OutputIntegration {
        input: our_file.to_string(),
        output: ours_bin.clone(),
        link: vec![helper_obj.clone()],
    });
    run_cli(&cli).unwrap();
    let ours_code = Command::new(&ours_bin)
        .status()
        .expect("failed to run our binary")
        .code()
        .expect("our binary was killed by a signal");

    // The reference: both halves compiled by clang.
    let ref_bin = path_of("reference");
    let status = Command::new("clang")
        .args(["-O1", "-w", our_file, clang_file, "-o", &ref_bin])
        .status()
        .expect("failed to start clang");
    assert!(status.success(), "clang failed to build the reference");
    let ref_code = Command::new(&ref_bin)
        .status()
        .expect("failed to run reference binary")
        .code()
        .expect("reference binary was killed by a signal");

    assert_eq!(ours_code, ref_code);
}

/// Like `test_c_file`, but our side runs as a SUBPROCESS binary instead
/// of in-process JIT: for fixtures whose failure mode is a crash (e.g.
/// a misaligned SSE store), a JIT run would take the whole test harness
/// down with it, while a crashed child is just a differing exit code.
fn test_c_file_binary(input: &str) {
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let ours_bin = tmp_dir
        .path()
        .join("ours")
        .to_str()
        .expect("failed to construct path")
        .to_string();
    let cli = Cli::get_output_integration(OutputIntegration {
        input: input.to_string(),
        output: ours_bin.clone(),
        link: vec![],
    });
    run_cli(&cli).unwrap();
    let ours_code = Command::new(&ours_bin)
        .status()
        .expect("failed to run our binary")
        .code();
    let clang_code = run_example_clang(input);
    assert_eq!(
        ours_code,
        Some(clang_code),
        "our binary exited {ours_code:?} (None = killed by signal), clang's exited {clang_code}"
    );
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

// Reduced from SQLite: a top-level (non-loop) branch whose continuations are a
// mix of in-region points and boundaries, which the restructure transform only
// handles inside a loop body (LoopBodyExit::Demux). Currently fails in
// structure_seq / structure_capture ("mixed in-region/boundary demux ...").
#[test]
fn test_38_mixed_continuation_demux() {
    test_c_file("tests/fixtures/c/38_mixed_continuation_demux.c");
}

// Mutually-referencing loop phis (a pure swap): the arc payload must apply
// phi copies as PARALLEL copies -- resolve every incoming against the scope
// as it stood before the arc, then write all destinations. Sequential
// resolve-and-write feeds the second phi the first one's new value.
#[test]
fn test_39_swap_phis() {
    test_c_file("tests/fixtures/c/39_swap_phis.c");
}

// Reduced from SQLite: a global struct initializer taking a function's
// address. Globals are lowered before function bodies, so the FuncAddr
// constant requires every function to be DECLARED (register_fn) before any
// global initializer is lowered.
#[test]
fn test_40_global_fn_ptr() {
    test_c_file("tests/fixtures/c/40_global_fn_ptr.c");
}

// Reduced from SQLite: LLVM's icmp accepts pointer operands
// (`icmp eq ptr %p, null`); the compare lowering must build a pointer
// compare rather than unwrap the operands as integers.
#[test]
fn test_41_ptr_compare() {
    test_c_file("tests/fixtures/c/41_ptr_compare.c");
}

// Reduced from SQLite: a function pointer crossing branch joins and a loop
// header (phi over function addresses). A global/function reference
// constant's value type must be a pointer, not the referent's type -- a
// function-typed loop slot has no LLVM value representation.
#[test]
fn test_42_fn_ptr_phi() {
    test_c_file("tests/fixtures/c/42_fn_ptr_phi.c");
}

// A plain global's address through branch joins and a loop header: the
// pointer-typed sibling of fixture 42. Guards the value type of GlobalAddr
// constants (a pointer, never the referent's type).
#[test]
fn test_43_global_addr_phi() {
    test_c_file("tests/fixtures/c/43_global_addr_phi.c");
}

// Atomic loads, stores, and a fence with explicit orderings; single
// threaded, so the differential checks the value flow while the orderings
// exercise the instruction attributes.
#[test]
fn test_44_atomic_load_store() {
    test_c_file("tests/fixtures/c/44_atomic_load_store.c");
}

// Atomic read-modify-write (fetch_add, exchange) and strong
// compare-and-swap, including a failing swap that writes the observed
// value back through `expected`. The pair result flows through
// extractvalue to the node's projections.
#[test]
fn test_45_atomic_rmw_cas() {
    test_c_file("tests/fixtures/c/45_atomic_rmw_cas.c");
}

// A thread-local global: the thread_local mode must survive re-emission
// (accesses go through llvm.threadlocal.address, whose operand LLVM
// requires to be thread-local).
#[test]
fn test_46_thread_local() {
    test_c_file("tests/fixtures/c/46_thread_local.c");
}

// Module-level inline assembly defining a real symbol; must be preserved
// verbatim or the symbol vanishes and linking fails.
#[test]
fn test_47_module_asm() {
    test_c_file("tests/fixtures/c/47_module_asm.c");
}

// Two-TU ABI differential, caller side: our main passes a 32-byte struct
// by value to a clang-compiled callee; the call site must carry byval.
#[test]
fn test_48_byval_call() {
    test_c_pair(
        "tests/fixtures/c/48_byval_call.c",
        "tests/fixtures/c/48_byval_call_helper.c",
    );
}

// Two-TU ABI differential, callee side: clang's main calls our function
// taking a 32-byte struct by value; our definition must carry byval.
#[test]
fn test_49_byval_callee() {
    test_c_pair(
        "tests/fixtures/c/49_byval_callee.c",
        "tests/fixtures/c/49_byval_callee_main.c",
    );
}

// Two-TU ABI differential, variadic caller side: our main passes a
// 32-byte struct through `...`; the call site is the only place that
// argument's byval attribute exists, so dropping call-site attributes on
// direct calls makes the clang-compiled callee's va_arg read garbage.
#[test]
fn test_53_variadic_byval() {
    test_c_pair(
        "tests/fixtures/c/53_variadic_byval.c",
        "tests/fixtures/c/53_variadic_byval_helper.c",
    );
}

// 54-64: accumulator-promotion corpus. Loops that read-modify-write an
// array cell, from the plainly promotable shapes through the aliasing,
// zero-trip, conditional, early-exit, varying-address and call-clobber
// hazards where the cell's memory traffic is semantically visible. Each
// fixture's comment describes the shape; the exit codes pin the
// semantics.
#[test]
fn test_54_promote_basic() {
    test_c_file("tests/fixtures/c/54_promote_basic.c");
}

#[test]
fn test_55_promote_gemm() {
    test_c_file("tests/fixtures/c/55_promote_gemm.c");
}

#[test]
fn test_56_promote_two_cells() {
    test_c_file("tests/fixtures/c/56_promote_two_cells.c");
}

#[test]
fn test_57_promote_local_array() {
    test_c_file("tests/fixtures/c/57_promote_local_array.c");
}

#[test]
fn test_58_promote_alias_same_array() {
    test_c_file("tests/fixtures/c/58_promote_alias_same_array.c");
}

#[test]
fn test_59_promote_alias_params() {
    test_c_file("tests/fixtures/c/59_promote_alias_params.c");
}

#[test]
fn test_60_promote_zero_trip() {
    test_c_file("tests/fixtures/c/60_promote_zero_trip.c");
}

#[test]
fn test_61_promote_conditional() {
    test_c_file("tests/fixtures/c/61_promote_conditional.c");
}

#[test]
fn test_62_promote_early_exit() {
    test_c_file("tests/fixtures/c/62_promote_early_exit.c");
}

#[test]
fn test_63_promote_varying_address() {
    test_c_file("tests/fixtures/c/63_promote_varying_address.c");
}

#[test]
fn test_64_promote_call_clobber() {
    test_c_file("tests/fixtures/c/64_promote_call_clobber.c");
}

// Constant getelementptr expressions whose source element type is not
// recoverable from the base pointer (element-typed pointer arithmetic,
// aggregate descent): the type must come from the expression itself.
#[test]
fn test_65_const_gep_addresses() {
    test_c_file("tests/fixtures/c/65_const_gep_addresses.c");
}

// An over-aligned local whose alloca alignment must survive emission:
// the constant-copy initialisation claims align 16 and the backend
// expands it with aligned SSE stores, so a dropped alloca alignment
// segfaults. Subprocess harness: the failure mode is a crash.
#[test]
fn test_66_alloca_alignment() {
    test_c_file_binary("tests/fixtures/c/66_alloca_alignment.c");
}

// Bitfield storage units have non-power-of-two widths (a 22-bit field
// stores as a 24-bit integer): the type and its masked accesses must
// flow through the whole pipeline.
#[test]
fn test_67_bitfield_i24() {
    test_c_file("tests/fixtures/c/67_bitfield_i24.c");
}

// An alignas(64) global: the alignment attribute must survive re-emission
// (checked through the global's runtime address).
#[test]
fn test_50_global_alignment() {
    test_c_file("tests/fixtures/c/50_global_alignment.c");
}
