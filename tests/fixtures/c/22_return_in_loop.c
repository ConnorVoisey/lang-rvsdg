// Tier 22 — early return inside a loop body. §4.1.
//
// A `return` from inside the loop should become an exit arc to the
// function-exit block, carrying its value like any other multi-exit path
// (the body walker has a `Ret`-in-loop rejection in
// src/llvm_parser/region/loops.rs `lower_body_walk`). In practice it
// currently fails even earlier, in dominance computation
// (src/llvm_parser/dominance.rs): the early return leaves a block
// unreachable in the post-dominator view and trips a precondition. csmith
// emits this routinely (a `return` from inside a loop).

int f(int n) {
    for (int i = 0; i < n; i++) {
        if (i == 3) return i * 2;
    }
    return -1;
}

int main(void) {
    return f(10); // returns at i==3 -> 6
}
