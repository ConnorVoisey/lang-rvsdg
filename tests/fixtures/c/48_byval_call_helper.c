// The clang-compiled side of the 48_byval_call ABI differential.

struct big {
    long a, b, c, d;
};

int sum_big(struct big v, int expected) {
    return (int)(v.a + v.b + v.c + v.d) - expected;
}
