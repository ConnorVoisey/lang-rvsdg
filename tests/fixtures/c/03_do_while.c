// Tier 3 — the simplest natural loop: do-while.
//
// After mem2reg, this collapses to a SINGLE-BLOCK SCC with a self-edge.
// Header == latch == exit-source, all the same block. The phi for `x`
// sits at the top, the body and the test are interleaved in the block,
// the conditional branch loops back or exits.
//
// This is the cleanest test of θ-node construction with NO inner γ:
// - One loop-variant value (x)
// - Body region is straight-line code
// - The conditional itself becomes the θ-node's repetition predicate
//
// `while`/`for` loops do NOT produce this shape without `loop-rotate`,
// because the test sits at the header in a separate block from the body.

int main() {
    int x = 0;
    do {
        x = x + 1;
    } while (x < 10);
    return x;
}
