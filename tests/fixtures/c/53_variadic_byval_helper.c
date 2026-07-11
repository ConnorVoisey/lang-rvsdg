// The clang-compiled side of the 53_variadic_byval ABI differential:
// reads a 32-byte struct and a trailing int back out of the va_list.
#include <stdarg.h>

struct big {
    long a, b, c, d;
};

int sum_variadic(int count, ...) {
    va_list ap;
    va_start(ap, count);
    struct big v = va_arg(ap, struct big);
    int expected = va_arg(ap, int);
    va_end(ap);
    return (int)(v.a + v.b + v.c + v.d) - expected;
}
