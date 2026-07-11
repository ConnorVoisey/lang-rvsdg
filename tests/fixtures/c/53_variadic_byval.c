// Two-TU ABI differential, variadic caller side: our compiled main passes
// a 32-byte struct through `...` to a clang-compiled variadic callee. A
// variadic call site is the ONLY place the struct argument's byval
// attribute exists (the declaration has no parameter entry for it): with
// byval the struct is copied to the caller's outgoing stack area, where
// the callee's va_arg reads it; if byval is dropped, only a pointer to
// the struct travels (in a register) and va_arg reads garbage from the
// empty overflow area. The trailing scalar stays in a register either
// way, pinning the register assignments. Linked against
// 53_variadic_byval_helper.c compiled by clang.

struct big {
    long a, b, c, d;
};

int sum_variadic(int count, ...);

int main(void) {
    struct big v = {10, 20, 30, 40};
    return sum_variadic(1, v, 58);
}
