//! The reparse-and-diff fidelity net: parse an input, run it through the
//! full pipeline (RVSDG construction and LLVM re-emission), REPARSE our
//! emitted module with llvm-ir, and diff the global/function headers
//! field by field against the input's. Both sides go through the same
//! parser, so a mismatch is a real drop or invention in our pipeline, not
//! a formatting difference.
//!
//! The diff has NO general allowlist: anything we deliberately do not
//! re-emit must be normalised here, one narrowly-scoped rule per line,
//! with the reason. Everything else failing is a fidelity gap.

use std::collections::BTreeSet;
use std::path::Path;

use inkwell::context::Context;
use lang_rvsdg::{c_file_to_mod, rvsdg::RVSDGMod};
use llvm_ir::Module;

/// Strip llvm-ir's name wrapper (its Display prepends a sigil).
fn name_of(name: &llvm_ir::Name) -> String {
    match name {
        llvm_ir::Name::Name(s) => s.as_ref().clone(),
        llvm_ir::Name::Number(n) => n.to_string(),
    }
}

fn parse_input(input: &str) -> Module {
    c_file_to_mod(Path::new(input), &[], &[], true)
        .unwrap_or_else(|e| panic!("failed to parse {input}: {e}"))
}

fn reparse_our_output(input: &str) -> Module {
    let module = parse_input(input);
    let rvsdg = RVSDGMod::from_llvm_mod(module)
        .unwrap_or_else(|e| panic!("construction failed for {input}: {e}"));
    let context = Context::create();
    let emitted = rvsdg
        .lower_to_llvm_module(&context)
        .unwrap_or_else(|e| panic!("lowering failed for {input}: {e}"));
    let text = emitted.print_to_string().to_string();
    Module::from_ir_str(&text)
        .unwrap_or_else(|e| panic!("our emitted module for {input} does not reparse: {e}"))
}

/// One comparison line; pushes a readable gap description on mismatch.
macro_rules! cmp_field {
    ($gaps:expr, $what:expr, $field:literal, $input:expr, $ours:expr) => {
        if $input != $ours {
            $gaps.push(format!(
                "{}: {}: input {:?}, ours {:?}",
                $what, $field, $input, $ours
            ));
        }
    };
}

/// Attribute lists compared as multisets of their Debug forms, order-blind.
fn attr_set<T: std::fmt::Debug>(attrs: &[T]) -> BTreeSet<String> {
    attrs.iter().map(|a| format!("{a:?}")).collect()
}

fn diff_attr_sets(
    gaps: &mut Vec<String>,
    what: &str,
    input: &BTreeSet<String>,
    ours: &BTreeSet<String>,
) {
    let dropped: Vec<_> = input.difference(ours).cloned().collect();
    let invented: Vec<_> = ours.difference(input).cloned().collect();
    if !dropped.is_empty() {
        gaps.push(format!("{what}: attributes dropped: {dropped:?}"));
    }
    if !invented.is_empty() {
        gaps.push(format!("{what}: attributes invented: {invented:?}"));
    }
}

fn diff_headers(input: &Module, ours: &Module) -> Vec<String> {
    let mut gaps = Vec::new();

    for g in &input.global_vars {
        let name = name_of(&g.name);
        let what = format!("global {name}");
        let Some(o) = ours.global_vars.iter().find(|o| name_of(&o.name) == name) else {
            gaps.push(format!("{what}: missing from our output"));
            continue;
        };
        cmp_field!(gaps, what, "linkage", g.linkage, o.linkage);
        cmp_field!(gaps, what, "visibility", g.visibility, o.visibility);
        cmp_field!(gaps, what, "is_constant", g.is_constant, o.is_constant);
        cmp_field!(gaps, what, "addr_space", g.addr_space, o.addr_space);
        cmp_field!(
            gaps,
            what,
            "dll_storage_class",
            g.dll_storage_class,
            o.dll_storage_class
        );
        cmp_field!(
            gaps,
            what,
            "thread_local_mode",
            g.thread_local_mode,
            o.thread_local_mode
        );
        cmp_field!(gaps, what, "unnamed_addr", g.unnamed_addr, o.unnamed_addr);
        cmp_field!(gaps, what, "section", g.section, o.section);
        // Alignment 0 on the input means "unspecified"; ours may then also
        // be unspecified or the ABI value LLVM assigns. Only compare when
        // the input pinned it.
        if g.alignment != 0 {
            cmp_field!(gaps, what, "alignment", g.alignment, o.alignment);
        }
        cmp_field!(
            gaps,
            what,
            "has_initializer",
            g.initializer.is_some(),
            o.initializer.is_some()
        );
    }

    for f in &input.functions {
        let what = format!("fn {}", f.name);
        let Some(o) = ours.functions.iter().find(|o| o.name == f.name) else {
            gaps.push(format!("{what}: missing from our output (as a definition)"));
            continue;
        };
        cmp_field!(gaps, what, "linkage", f.linkage, o.linkage);
        cmp_field!(gaps, what, "visibility", f.visibility, o.visibility);
        cmp_field!(
            gaps,
            what,
            "dll_storage_class",
            f.dll_storage_class,
            o.dll_storage_class
        );
        cmp_field!(
            gaps,
            what,
            "calling_convention",
            f.calling_convention,
            o.calling_convention
        );
        cmp_field!(gaps, what, "is_var_arg", f.is_var_arg, o.is_var_arg);
        cmp_field!(gaps, what, "section", f.section, o.section);
        if f.alignment != 0 {
            cmp_field!(gaps, what, "alignment", f.alignment, o.alignment);
        }
        cmp_field!(
            gaps,
            what,
            "gc",
            f.garbage_collector_name,
            o.garbage_collector_name
        );
        cmp_field!(
            gaps,
            what,
            "has_personality",
            f.personality_function.is_some(),
            o.personality_function.is_some()
        );
        diff_attr_sets(
            &mut gaps,
            &format!("{what}: function_attributes"),
            &attr_set(&f.function_attributes),
            &attr_set(&o.function_attributes),
        );
        diff_attr_sets(
            &mut gaps,
            &format!("{what}: return_attributes"),
            &attr_set(&f.return_attributes),
            &attr_set(&o.return_attributes),
        );
        for (i, (ip, op)) in f.parameters.iter().zip(o.parameters.iter()).enumerate() {
            diff_attr_sets(
                &mut gaps,
                &format!("{what}: param {i}"),
                &attr_set(&ip.attributes),
                &attr_set(&op.attributes),
            );
        }
        if f.parameters.len() != o.parameters.len() {
            gaps.push(format!(
                "{what}: parameter count: input {}, ours {}",
                f.parameters.len(),
                o.parameters.len()
            ));
        }
    }

    for d in &input.func_declarations {
        let what = format!("declare {}", d.name);
        let Some(o) = ours.func_declarations.iter().find(|o| o.name == d.name) else {
            gaps.push(format!("{what}: missing from our output"));
            continue;
        };
        cmp_field!(gaps, what, "linkage", d.linkage, o.linkage);
        cmp_field!(
            gaps,
            what,
            "calling_convention",
            d.calling_convention,
            o.calling_convention
        );
        diff_attr_sets(
            &mut gaps,
            &format!("{what}: return_attributes"),
            &attr_set(&d.return_attributes),
            &attr_set(&o.return_attributes),
        );
        for (i, (ip, op)) in d.parameters.iter().zip(o.parameters.iter()).enumerate() {
            diff_attr_sets(
                &mut gaps,
                &format!("{what}: param {i}"),
                &attr_set(&ip.attributes),
                &attr_set(&op.attributes),
            );
        }
    }

    gaps
}

fn assert_header_fidelity(input: &str) {
    let parsed_input = parse_input(input);
    let ours = reparse_our_output(input);
    let gaps = diff_headers(&parsed_input, &ours);
    assert!(
        gaps.is_empty(),
        "fidelity gaps in {input} ({} total):\n  {}",
        gaps.len(),
        gaps.join("\n  ")
    );
}

// The baseline: even a trivial program carries function attributes (clang
// stamps target-cpu/target-features/frame-pointer string attributes and
// noundef on every function).
#[test]
fn fidelity_00_straight_line() {
    assert_header_fidelity("tests/fixtures/c/00_straight_line.c");
}

// Optimisation-hint parameter attributes: restrict (noalias) and the
// noundef clang adds everywhere.
#[test]
fn fidelity_51_param_hints() {
    assert_header_fidelity("tests/fixtures/c/51_attr_param_hints.c");
}

// Memory-effect function attributes: __attribute__((const)) is
// memory(none), __attribute__((pure)) is memory(read).
#[test]
fn fidelity_52_fn_memory() {
    assert_header_fidelity("tests/fixtures/c/52_attr_fn_memory.c");
}

// The globals and atomics fixtures cover the attribute-heavy shapes the
// real-world builds needed.
#[test]
fn fidelity_50_global_alignment() {
    assert_header_fidelity("tests/fixtures/c/50_global_alignment.c");
}

#[test]
fn fidelity_46_thread_local() {
    assert_header_fidelity("tests/fixtures/c/46_thread_local.c");
}

// The whole fixture corpus: every attribute any fixture exercises must
// survive the round trip. New fixtures are covered automatically.
#[test]
fn fidelity_all_fixtures() {
    let mut checked = 0;
    for entry in std::fs::read_dir("tests/fixtures/c").expect("fixture dir") {
        let path = entry.expect("fixture entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("c") {
            continue;
        }
        assert_header_fidelity(path.to_str().expect("fixture path"));
        checked += 1;
    }
    assert!(checked > 50, "expected the full corpus, found {checked}");
}
