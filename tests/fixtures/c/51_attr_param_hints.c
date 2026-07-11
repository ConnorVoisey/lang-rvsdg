// Optimisation-hint parameter attributes: restrict becomes noalias on the
// definition, and clang stamps noundef on every parameter. These do not
// change the ABI, but dropping them costs the eventual RVSDG optimiser
// alias information and breaks header fidelity.

static int combine(int *restrict a, int *restrict b) { return *a * 10 + *b; }

int main(void) {
    int x = 3;
    int y = 4;
    return combine(&x, &y) - 34;
}
