// Two-TU ABI differential, caller side: our compiled main passes a
// 32-byte struct BY VALUE plus a scalar to a clang-compiled callee. The
// byval attribute moves the struct to the stack, so the scalar lands in
// the FIRST integer register; dropping byval shifts every register
// assignment (struct pointer takes the first register, scalar the second)
// and the callee deterministically reads the wrong values -- no reliance
// on stack-layout luck. Linked against 48_byval_call_helper.c compiled by
// clang.

struct big {
    long a, b, c, d;
};

int sum_big(struct big v, int expected);

int main(void) {
    struct big v = {10, 20, 30, 40};
    return sum_big(v, 100);
}
