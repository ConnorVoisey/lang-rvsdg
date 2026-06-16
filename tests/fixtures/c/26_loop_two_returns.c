// Tier 26 — loop with two early returns plus a natural exit.
//
// Three exits that land in different places: two `return` exits (to the
// function exit) and the fall-through after the loop. Stresses multi-exit
// dispatch where the exit targets don't all reconverge at one block before
// the function exit.

int f(int n) {
    for (int i = 0; i < n; i++) {
        if (i == 3) return 100;
        if (i == 7) return 200;
    }
    return -1;
}

int main(void) {
    return f(5); // hits i==3 -> 100
}
