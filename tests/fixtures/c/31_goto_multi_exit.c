// Tier 31 — loop body `goto`s to two DIFFERENT labels outside the loop.
//
// The two exit targets (`hot` and `cold`) never reconverge before the
// function exit — each does different work and returns. This is the
// canonical "exit targets have no common post-dominator" shape that needs
// the §4.2 continuation predicate on the post-loop dispatch.

int f(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (i == 3) goto hot;
        if (i == 6) goto cold;
        s += i;
    }
    return s;
hot:
    return s * 2;
cold:
    return s + 1;
}

int main(void) {
    return f(10); // i==3 -> hot -> (0+1+2)*2 = 6
}
