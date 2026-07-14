// Accumulator whose base is a LOCAL array (stack storage, not a
// global): acc[i] is address-invariant in the j loop, and the outer
// loop accumulates into a different cell of the same base each
// iteration.

int table[8][8];

int main(void) {
    int acc[8] = {0};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            table[i][j] = (i * j + 3) % 11;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            acc[i] += table[i][j];
    return (acc[3] + acc[5]) % 256;
}
