// Accumulator cell with a loop-invariant address, the simplest
// promotable shape: c[3] is read and written every iteration while the
// summed array is a distinct global. The exit code pins the final value.

double a[64], c[8];

int main(void) {
    for (int i = 0; i < 64; i++)
        a[i] = i % 7;
    c[3] = 1.0;
    for (int i = 0; i < 64; i++)
        c[3] += a[i];
    return (int)c[3] % 256;
}

// double a[64], c[8];
//
// int main(void) {
//     for (int i = 0; i < 64; i++)
//         a[i] = i % 7;
//     double local_c_3 = 1.0;
//     for (int i = 0; i < 64; i++)
//         local_c_3 += a[i];
//     return (int)local_c_3 % 256;
// }
