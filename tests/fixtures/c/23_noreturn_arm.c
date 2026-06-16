// Tier 23 — one branch arm does not return (noreturn call). §4.2.
//
// `abort()` is `noreturn`, so clang emits `call abort(); unreachable` in
// the then-arm. That arm never reaches the function exit — the genuine
// `p`-predicate territory: a continuation that goes "nowhere" (⊥).
//
// Currently fails earlier than the branch handling: the post-dominator
// traversal (src/llvm_parser/dominance.rs) DFS-visits only blocks that can
// reach the exit, so the dead arm trips a block-count precondition
// (dominance.rs:90). Same blocker as tier 22; fixing post-dominance for
// exit-unreachable blocks unblocks both. Only then does the §4.2
// no-common-post-dominator path matter.

#include <stdlib.h>

int f(int x) {
    if (x < 0) abort();
    return x + 1;
}

int main(void) {
    return f(5); // 5 >= 0, no abort -> 6
}
