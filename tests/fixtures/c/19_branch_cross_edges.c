// Tier 19 — branch arms reaching a SHARED continuation point (cross
// edges; paper Fig. 4, "inconsistent selection paths"). §4.2.
//
// `Y` is reachable from two different arms (the inner `else` and the
// outer `else`), so it is dominated by neither. Handled by the path-aware
// walk: each arm that reaches `Y` lowers it independently and binds the
// final join phi from the block it exited through, rather than from a
// static dominator set. Uses `goto`, which csmith emits under `--jumps`.
//
// (Both arms still reconverge at the function exit, so a single
// post-dominator join exists; the separate §4.2 `p` dispatch is only for
// branches with no common post-dominator at all.)

int f(int a, int b) {
    int r;
    if (a) {
        if (b) goto X;
        else goto Y;
    } else {
        goto Y;
    }
X:
    r = 1;
    goto END;
Y:
    r = 2;
    goto END;
END:
    return r;
}

int main(void) {
    return f(1, 0); // a=1, b=0 -> Y -> 2
}
