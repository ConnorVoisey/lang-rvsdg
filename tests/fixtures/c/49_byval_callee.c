// Two-TU ABI differential, callee side: OUR compiled function takes a
// 32-byte struct by value plus a scalar, called from a clang-compiled
// main. Our emitted definition must carry byval or every register
// assignment shifts (see 48_byval_call.c for the mechanism) and the
// function deterministically computes garbage. Linked against
// 49_byval_callee_main.c compiled by clang.

struct big {
    long a, b, c, d;
};

int sum_big(struct big v, int expected) {
    return (int)(v.a + v.b + v.c + v.d) - expected;
}
