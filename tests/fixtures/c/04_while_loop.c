// Tier 4 — natural loop with an internal branch (while/for shape).
//
// Now the SCC has TWO blocks (header + latch). The header tests `i < 5`
// and either falls into the body or exits. The body falls back to header.
//
// LoopArcs reports it as Natural (1 entry, 1 back-edge, 1 exit), so a
// θ-node is the right wrapper — but inside the θ body, the conditional
// at the header creates an internal control-flow split. That needs a
// γ-node nested inside the θ-node:
//
//   θ {
//     body region:
//       cond = (i < 5)
//       γ on cond {
//         arm true:  ... body work ... compute next i
//         arm false: produce dummy/exit values
//       }
//       predicate = cond
//   }
//
// This is the first example that requires Phase 3 (branch restructuring)
// to handle correctly — the γ-arms have to converge before the θ's
// repetition predicate is decided.

int main() {
    int sum = 0;
    int i = 0;
    while (i < 5) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}
