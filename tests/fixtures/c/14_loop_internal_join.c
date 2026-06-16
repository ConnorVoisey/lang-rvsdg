// Tier 14 - loop with an internal if/then producing a join phi that
// later body instructions use. This shape is what mem2reg+lcssa
// produces from any C loop containing `if (...) { x = ...; }` with no
// matching else: a join block at the top of the if's continuation
// holds `phi i32 [ %original, %cond_block ], [ %updated, %if_true ]`,
// and the rest of the body consumes the phi's destination.
//
// Without path-aware phi resolution in the body walker, the join phi
// is skipped by lower_instructions_skip_phis and the next instruction
// that references the destination panics in operand() with
// "ssa value should already have been defined".

int main() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        int x = i;
        if (i % 2 == 0) {
            x = i * 2;
        }
        sum += x;
    }
    return sum;
}
