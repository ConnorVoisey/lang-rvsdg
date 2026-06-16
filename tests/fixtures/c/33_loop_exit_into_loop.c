// Tier 33 — a loop exit that lands inside another loop's body (not at its
// preheader): the outer loop contains an inner loop, and the inner loop's
// break continues the OUTER loop directly.
//
// Distinct from tier 29 (sequential loops) and tier 21 (break out of both):
// here the inner exit resumes mid-outer-body.

int main(void) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (j == i) break; // exit inner, continue outer body
            s += 1;
        }
        s += 10; // outer-body work after the inner loop
    }
    return s;
}
