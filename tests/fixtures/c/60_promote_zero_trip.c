// The accumulator pointer is NULL on a call whose loop runs zero times:
// the cell is only ever touched inside the loop body, so the program is
// well defined. Reading or writing the cell outside the trip-count
// guard dereferences NULL and crashes.

double cells[4];

int accumulate(double *cell, const double *vals, int n) {
    for (int i = 0; i < n; i++)
        *cell += vals[i];
    return n;
}

int main(void) {
    double vals[4] = {1.0, 2.0, 3.0, 4.0};
    int ran = accumulate(&cells[1], vals, 4);
    ran += accumulate(0, vals, 0);
    return (int)cells[1] + ran;
}
