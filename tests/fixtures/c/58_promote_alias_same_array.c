// The loop sweeps a store across the SAME array that holds the
// accumulator cell: at i == 5 the a[i] store overwrites a[5] mid-loop,
// so the accumulator's memory value is not private to the accumulation.
// Treating a[5] as a carried register for the whole loop loses the
// overwrite and changes the result (55 with the overwrite, 101 without).

int a[10];

int main(void) {
    a[5] = 1;
    for (int i = 0; i < 10; i++) {
        a[i] = i;
        a[5] += 10;
    }
    return a[5];
}
