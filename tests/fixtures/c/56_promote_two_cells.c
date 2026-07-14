// Two independent accumulator cells in the same loop body, plus a
// streaming store to a third array at a varying index. The second
// accumulator reads the freshly streamed value, so the ordering between
// the dst store and the acc[2] update is semantically visible; the exit
// code observes both accumulators and a streamed element.

int src[32], dst[32], acc[4];

int main(void) {
    for (int i = 0; i < 32; i++)
        src[i] = i ^ 5;
    for (int i = 0; i < 32; i++) {
        dst[i] = src[i] * 2;
        acc[1] += src[i];
        acc[2] += dst[i];
    }
    return (acc[1] + acc[2] + dst[7]) % 256;
}
