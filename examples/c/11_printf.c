// Probe: does variadic printf survive the round-trip through opt -> our
// parser -> RVSDG -> codegen -> JIT? csmith uses printf for its hash
// dump, so this is a prerequisite for csmith-style differential testing.

#include <stdio.h>

int main() {
    int a = 42;
    int b = 7;
    printf("hello %d %d\n", a, b);
    return 0;
}
