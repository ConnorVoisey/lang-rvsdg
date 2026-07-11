// The clang-compiled side of the 49_byval_callee ABI differential.

struct big {
    long a, b, c, d;
};

int sum_big(struct big v, int expected);

int main(void) {
    struct big v = {1, 2, 3, 4};
    return sum_big(v, 10);
}
