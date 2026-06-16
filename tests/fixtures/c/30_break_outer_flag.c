// Tier 30 — break out of nested loops without goto, via a flag the outer
// loop checks.
//
// The inner `break` exits the inner loop; the outer loop then tests the
// flag and breaks too. Two structured breaks at different levels, both
// reconverging after the outer loop.

int main(void) {
    int s = 0;
    int done = 0;
    for (int i = 0; i < 5 && !done; i++) {
        for (int j = 0; j < 5; j++) {
            if (i + j == 4) {
                done = 1;
                break;
            }
            s += 1;
        }
    }
    return s; // increments until i+j==4
}
