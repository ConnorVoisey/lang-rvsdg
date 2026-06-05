// Tier 7 -- triple-nested loops.
//
// Three loops nested three deep. After mem2reg + loop-rotate + lcssa each
// becomes a distinct natural loop, with the parent chain
//   L_k (innermost) -> L_j -> L_i.
//
// Exercises:
// - LoopForest computes a three-level nesting relation.
// - Each inner-loop dest threads through TWO surrounding theta nodes:
//   k's body increments `total`, then leaves k's theta, then j's body
//   continues with the incremented total, leaves j's theta, then i's
//   body increments i and so on.
// - The innermost theta produces a value (total updated) that becomes
//   the next-iteration value of the middle theta's `total` loop_var,
//   which in turn feeds the outer theta's `total` loop_var.

int main() {
    int total = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                total = total + 1;
            }
        }
    }
    return total; // 2 * 3 * 4 = 24
}
