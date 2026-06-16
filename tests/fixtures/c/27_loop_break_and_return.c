// Tier 27 — loop with a `break` AND a `return` inside the body.
//
// Mixed exit kinds: the `break` exits to the after-loop block, the
// `return` exits to the function exit. The two exit targets do not
// reconverge until the function exit, so the post-loop dispatch must cope
// with non-reconverging exit targets.

int f(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (i == 2) return s + 1000;
        if (i == 4) break;
        s += i;
    }
    return s;
}

int main(void) {
    return f(10); // i==2 -> return 0+1000 = 1000
}
