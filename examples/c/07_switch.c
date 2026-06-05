// Tier 7 — switch statement (multi-arm γ-node).
//
// switch is a control-flow split with N targets instead of 2. In RVSDG,
// it becomes an N-arm γ-node (γ_n with one region per case), with the
// switch's value as the predicate.
//
// No new structural complexity beyond the if/else case (Tier 2) — just a
// wider γ-node. But it's worth a separate test because:
// - The terminator handling code now sees `Terminator::Switch` instead
//   of `CondBr`, with a list of cases plus a default.
// - The γ_n builder takes &[&dyn Fn(...)] for its arms; you have to
//   construct the right arm count and order.
// - All arms have to converge (have the same shape of region results)
//   for the post-switch code to use them via phi nodes.

int classify(int x) {
    switch (x) {
        case 0: return 100;
        case 1: return 200;
        case 2: return 300;
        default: return 999;
    }
}

int main() {
    return classify(1);
}
