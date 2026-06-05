// Tier 7 -- loop nested inside a gamma arm.
//
// The natural-loop forest detects the inner loop. lower_region's gamma
// lowering walks each arm; the arm that contains the loop must dispatch
// to lower_loop when it reaches the loop header (and resume at the
// loop's single exit) before joining the gamma's post-dominator.
//
// `compute` is a separate function so the LLVM `-O1 -disable-llvm-passes
// + sroa,mem2reg,...` pipeline doesn't constant-fold the gamma away.
// The gamma branches on `x > 0`, which only becomes known at the call
// site after inlining (and we don't run inlining).

int compute(int x) {
    int sum = 0;
    if (x > 0) {
        for (int i = 0; i < x; i++) {
            sum = sum + i;
        }
    } else {
        sum = -1;
    }
    return sum;
}

int main() {
    return compute(5); // 0 + 1 + 2 + 3 + 4 = 10
}
