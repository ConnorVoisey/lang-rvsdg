// Tier 5 — loop with an early break: multi-exit.
//
// `break` introduces a SECOND exit arc from the loop body to outside.
// Now the SCC has 1 entry, 1 back-edge, but TWO exit arcs:
//   - The natural exit (i >= 100)
//   - The break exit (sum > 50)
// LoopArcs::is_natural() is false; the variant is Irregular.
//
// This is the first case that needs Phase 2's q/r restructuring, OR
// equivalent γ-dispatch logic in Phase 4 emission. The post-loop code
// must select between the two original exit blocks based on which
// path was taken — that's a γ-node on `q` after the θ-node.

int main() {
    int sum = 0;
    int i = 0;
    while (i < 100) {
        sum = sum + i;
        if (sum > 50) {
            break;
        }
        i = i + 1;
    }
    return sum;
}
