// Tier 35 — multi-exit loop where one exit target is `unreachable`.
//
// Minimal reduction (via llvm-reduce) of a csmith finding. The loop has two
// exits that stay distinct through loop-simplify: one returns (reaches the
// function exit), the other lands on a `__builtin_unreachable()` block that
// cannot reach the exit. The post-loop dispatch's join computation
// (`post_dominator_lca`, src/llvm_parser/region/loops.rs) finds no common
// post-dominator between the two exit targets and bails:
//   "multi-exit loop's exit targets … have no common post-dominator".
//
// The fix: an exit-unreachable exit target has no continuation, so it
// shouldn't participate in the join — the dispatch routes to it (it traps)
// and only the reachable exits need to reconverge.

int f(int x, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (s > 100) return s; // exit 1: returns
        switch (x) {
            case 0:
                s += i;
                break;
            case 1:
                s += 2 * i;
                break;
            default:
                __builtin_unreachable(); // exit 2: unreachable (no continuation)
        }
    }
    return s;
}

int main(void) {
    return f(0, 10); // x==0: s = 0+1+..+9 = 45
}
