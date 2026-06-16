// Tier 34 — a loop with two exits, one of which leads to an infinite loop
// (a block that never reaches the function exit).
//
// The two exit targets have no common post-dominator: the `done` path
// returns, but the `spin` path never reaches the exit, so the post-loop
// dispatch can't find a join. This is the shape that trips
// `post_dominator_lca` ("exit targets … have no common post-dominator").

int f(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (i == 3) goto done;
        if (i == 100) goto spin; // unreachable at runtime, but present statically
        s += i;
    }
done:
    return s;
spin:
    for (;;) {
    }
}

int main(void) {
    return f(10); // i==3 -> done -> 0+1+2 = 3
}
