// Atomic read-modify-write and compare-and-swap (single-threaded). The
// compare-and-swap pair result is consumed through extractvalue, which the
// frontend routes to the node's projections; the failing swap also checks
// that the old value is written back through `expected`.
#include <stdatomic.h>
#include <stdbool.h>

static _Atomic int value;

int main(void) {
    atomic_store(&value, 10);
    int old = atomic_fetch_add(&value, 5);
    int prev = atomic_exchange(&value, 42);
    int expected = 42;
    bool swapped = atomic_compare_exchange_strong(&value, &expected, 7);
    int expected2 = 100;
    bool failed = atomic_compare_exchange_strong(&value, &expected2, 9);
    int fin = atomic_load(&value);
    // old=10, prev=15, swapped=true, failed=false, expected2=7, fin=7.
    return (old - 10) + (prev - 15) + (swapped ? 0 : 1) + (failed ? 2 : 0)
        + (expected2 - 7) + (fin - 7);
}
