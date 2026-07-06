// Tier 39 -- mutually-referencing loop phis (the parallel-copy shape).
//
// The pure swap gives the loop header two phis whose back-edge incomings
// reference EACH OTHER's destinations: after mem2reg,
//   %a = phi [1, entry], [%b, latch]
//   %b = phi [2, entry], [%a, latch]
// where each incoming means the previous iteration's value. Binding the
// phi copies sequentially against the symbol table corrupts this (the
// second phi would read the first one's freshly-written value); the arc
// payload must resolve every incoming first and write all destinations
// after -- a parallel copy.

int f(int n) {
    int a = 1, b = 2;
    for (int i = 0; i < n; i++) {
        int t = a;
        a = b;
        b = t;
    }
    return a * 10 + b;
}

int main(void) {
    return f(3); // three swaps: (1,2) -> (2,1) -> (1,2) -> (2,1) => 21
}
