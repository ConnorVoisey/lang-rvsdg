// An over-aligned local initialised from a constant: the array gets
// `align 16` on its alloca and a copy from a constant claiming
// `ptr align 16`. The alloca's alignment must survive to emission --
// if the slot is laid out at the type's ABI alignment instead, the
// backend's aligned-SSE expansion of the copy faults. The odd-sized
// neighbouring local (360 bytes, 8 mod 16) shifts the frame so the
// misalignment is deterministic rather than layout luck; it is stored
// through so no frontend cleanup can delete it.

int main(void) {
    void *grid[5][3][3];
    long vals[4] = {5, 5, 5, 5};
    grid[0][0][0] = &vals[1];
    return (int)(vals[0] * 10 + vals[3] + (grid[0][0][0] != 0));
}
