// The accumulator cell is updated only on data-dependent iterations:
// the read-modify-write sits inside a branch, so the carried value must
// merge the updated and untouched paths every iteration.

int data[16], hist[2];

int main(void) {
    for (int i = 0; i < 16; i++)
        data[i] = i * 3 % 8;
    for (int i = 0; i < 16; i++) {
        if (data[i] > 3)
            hist[1] += data[i];
    }
    return hist[1];
}
