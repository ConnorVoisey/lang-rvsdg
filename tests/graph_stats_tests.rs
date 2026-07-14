//! Pins the census against real construction output (the demux-gamma
//! loop shape, constant-pool addresses, loop-var redirection), so
//! census refactors cannot silently change metric definitions.

use std::path::Path;

use lang_rvsdg::{c_file_to_mod, rvsdg::RVSDGMod, stats};

fn census_of(path: &str) -> stats::ModuleCensus {
    let module = c_file_to_mod(Path::new(path), &[], &[], true).unwrap();
    let rvsdg = RVSDGMod::from_llvm_mod(module).unwrap();
    assert!(rvsdg.verify().is_empty());
    stats::collect(&rvsdg)
}

/// Fixture 54: the second loop accumulates into c[3] (a constant-pool
/// address) while sweeping a distinct global -- one candidate; the
/// first loop stores a[i] at a varying address -- one varying bail.
/// Both loops sit behind the construction's continue/exit demux, so
/// this pins the redirection-aware invariance analysis end to end.
#[test]
fn census_of_promote_basic_fixture() {
    let census = census_of("tests/fixtures/c/54_promote_basic.c");
    let f = census
        .functions
        .iter()
        .find(|f| f.name == "main")
        .expect("main censused");
    assert_eq!(f.thetas, 2, "{f:?}");
    assert_eq!(f.promotion_candidates, 1, "{f:?}");
    assert_eq!(f.bail_varying, 1, "{f:?}");
    assert_eq!(
        f.bail_call + f.bail_alias + f.bail_nested + f.bail_sync,
        0,
        "{f:?}"
    );
}

/// Fixture 55 (loop-nest accumulator over globals): exactly one
/// candidate, the C[i][j] store promoted out of the innermost k loop;
/// the init and outer-level stores vary. The A/B loads must NOT count
/// as aliasing (distinct globals).
#[test]
fn census_of_promote_gemm_fixture() {
    let census = census_of("tests/fixtures/c/55_promote_gemm.c");
    let f = census
        .functions
        .iter()
        .find(|f| f.name == "main")
        .expect("main censused");
    assert_eq!(f.promotion_candidates, 1, "{f:?}");
    assert_eq!(f.bail_alias, 0, "{f:?}");
    assert!(f.licm_movable > 0, "{f:?}");
    // The innermost k loop carries i and j through unchanged, so the
    // loop nest must show pass-through freight. (Gamma outputs, by
    // contrast, are arm-value exports under this construction --
    // unchanged values bypass the demux as direct body params -- so no
    // gamma pass-through is expected here.)
    assert!(f.theta_passthrough > 0, "{f:?}");
}
