// Tier 0 — no control flow at all.
//
// The whole function is a single basic block ending in `ret`.
// Nothing for SCC analysis to find; no γ, no θ.
// Tests: instruction lowering for arithmetic + the function-result wiring.

int main() {
    int a = 3;
    int b = 4;
    int c = a + b;
    return c;
}
