// An over-aligned global: alignas(64) must survive re-emission or the
// global lands at its ABI alignment and the address check fails. The
// preceding one-byte global forces the array off 64-byte alignment unless
// the attribute pads it back; both are read through volatile so nothing
// folds away.

#include <stdint.h>

static char pad = 'x';
_Alignas(64) static long counters[4] = {1, 2, 3, 4};

int main(void) {
    char *volatile pp = &pad;
    long *volatile p = counters;
    int misaligned = ((uintptr_t)p % 64) != 0;
    return misaligned + (*pp - 'x') + (int)(p[0] + p[1] + p[2] + p[3]) - 10;
}
