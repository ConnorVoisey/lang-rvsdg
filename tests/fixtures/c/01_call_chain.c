// Tier 1 — straight-line with side-effecting calls.
//
// Still a single block, but now the state edge gets exercised:
// each `printf` consumes the function's state and produces a new one.
// Tests: state threading through a sequence of calls; constant string
// globals (the format strings become @.str globals).

#include <stdio.h>

int main() {
    int x = 10;
    printf("x is %d\n", x);
    printf("doubled is %d\n", x * 2);
    return 0;
}
