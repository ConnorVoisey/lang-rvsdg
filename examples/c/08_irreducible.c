// Tier 8 — irreducible loop via goto (multiple entry points).
//
// `goto label_inside_loop` from outside an existing cycle creates an
// SCC with TWO entry vertices. There's no single "header" — control
// can land at either entry depending on which path got taken.
//
// LoopArcs reports Irregular with entry_blocks.len() > 1. Phase 4 has to
// emit a γ-dispatch on the auxiliary `q` predicate INSIDE the θ body so
// each iteration starts at the right "entry slot".
//
// This is the canonical Bahmann §4.1 q/r restructuring case — natural
// language can't express it cleanly, only goto can.
//
// Compile with -O0 -Xclang -disable-llvm-passes; if loop-rotate or
// loop-simplify run, they may regularize this away.

int main() {
    int x = 0;
    int n = 5;

    if (n > 0) {
        goto inside;
    }

start:
    x = x + 10;

inside:
    x = x + 1;
    n = n - 1;
    if (n > 0) {
        goto start;
    }

    return x;
}
