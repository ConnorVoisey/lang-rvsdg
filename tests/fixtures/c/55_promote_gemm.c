// Loop-nest accumulator, the polybench shape: C[i][j]'s address is
// invariant only in the innermost (k) loop and varies with j one level
// up; the A and B loads vary with k. The exit code observes several
// cells of the result matrix.

int A[4][4], B[4][4], C[4][4];

int main(void) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            A[i][j] = i + j;
            B[i][j] = i * j + 1;
            C[i][j] = 0;
        }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                C[i][j] += A[i][k] * B[k][j];
    int sum = 0;
    for (int i = 0; i < 4; i++)
        sum += C[i][i] + C[i][3 - i];
    return sum % 256;
}
