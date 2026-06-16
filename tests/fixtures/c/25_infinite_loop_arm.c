// Tier 25 — one branch arm is an infinite loop. §4.2 / §4.1 boundary.
//
// `for(;;){}` in one arm is an SCC with no exit arc, so that arm never
// reaches the function exit. The enclosing branch therefore has no common
// post-dominator. Like tiers 22/23/24 it currently trips the
// post-dominator block-count precondition (dominance.rs:90), since the
// loop's blocks can't reach the exit in the reverse CFG. Beyond that it
// also stresses §4.1: a loop with zero exit arcs (a theta that never
// repeats out).

int f(int x) {
    if (x == 0) {
        for (;;) {
        }
    }
    return x;
}

int main(void) {
    return f(7); // 7 != 0, skip the loop -> 7
}
