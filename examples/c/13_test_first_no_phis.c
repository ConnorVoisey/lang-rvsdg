// Probe: test-first loop with no loop-carried scalar values. The
// condition reads a global, the body writes the global; after mem2reg
// there are no header phis. Exercises the gating gamma with an empty
// header_phis list (theta has zero loop variables apart from state).

int counter = 5;
int total = 0;

int main() {
    while (counter > 0) {
        total = total + counter;
        counter = counter - 1;
    }
    return total; // expected: 5 + 4 + 3 + 2 + 1 = 15
}
