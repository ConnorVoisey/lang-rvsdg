// Tier 2 — single if/else, no loop.
//
// Two basic blocks fork from a conditional branch and rejoin at a phi.
// SCC analysis sees three trivial SCCs (one per block); no cycles.
// The phi at the join is what RVSDG models as a γ-node — its result
// is the θ-free analogue of the loop case.
// Tests: γ-node construction, phi-as-region-output binding.

int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

int main() {
    return max(7, 3);
}
