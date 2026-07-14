// The accumulator pointer overlaps the source array: dst == &src[2], so
// iteration i == 2 reads the partially-accumulated value back through
// src[i]. Assuming the two pointers address disjoint memory changes the
// result (42 with the overlap honoured, 39 without).

int buf[8];

int accumulate(int *dst, int *src, int n) {
    for (int i = 0; i < n; i++)
        *dst += src[i];
    return *dst;
}

int main(void) {
    for (int i = 0; i < 8; i++)
        buf[i] = i + 1;
    return accumulate(&buf[2], buf, 8);
}
