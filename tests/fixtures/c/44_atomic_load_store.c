// Atomic loads, stores, and a fence (single-threaded). The exit-code
// differential checks the VALUES flow through correctly; the explicit
// orderings exercise the atomic attributes on the emitted instructions.
#include <stdatomic.h>

static _Atomic int counter;

int main(void) {
    atomic_store_explicit(&counter, 5, memory_order_relaxed);
    int a = atomic_load_explicit(&counter, memory_order_acquire);
    atomic_store_explicit(&counter, a + 2, memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);
    int b = atomic_load_explicit(&counter, memory_order_seq_cst);
    return b - 7;
}
