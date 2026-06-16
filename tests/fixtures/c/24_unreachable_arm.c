// Tier 24 — one branch arm is `__builtin_unreachable()`. §4.2.
//
// Like tier 23 but the dead arm is an explicit `unreachable` with no
// preceding call. Same `p`-predicate / no-common-post-dominator shape, and
// the same actual blocker: the then-arm can't reach the exit, so the
// post-dominator traversal trips the block-count precondition
// (dominance.rs:90). Shared blocker with tiers 22/23.

int f(int x) {
    if (x == 0) __builtin_unreachable();
    return 100 / x;
}

int main(void) {
    return f(4); // 100 / 4 = 25
}
