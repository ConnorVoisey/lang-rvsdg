// Tier 6 — nested loops.
//
// Two SCCs in the function: the inner loop is its own SCC, contained
// inside the outer loop's SCC. Tarjan's reverse-topological order means
// the inner SCC is processed FIRST and becomes a θ-node before the outer
// SCC sees it.
//
// From the outer SCC's perspective, the inner θ-node is opaque: it has
// one input (whatever flows in), one output (whatever flows out). The
// outer loop's body just contains "linear instructions, then a θ-node,
// then more linear instructions, then iterate."
//
// New things this exercises:
// - Recursive Phase 4: lower the inner θ-node, then continue lowering
//   the outer body using the inner θ's outputs.
// - A loop-variant value (`total`) that's modified BOTH directly in the
//   outer loop AND by the inner loop's accumulation — meaning the inner
//   θ-node's output threads into the outer θ-node's next-iteration value.

int main() {
    int total = 0;
    int i = 0;
    while (i < 3) {
        int j = 0;
        while (j < 4) {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
