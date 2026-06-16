// Tier 36 — global struct with a constant struct initializer.
//
// `struct S g = { ... }` lowers to a global whose initializer is a
// constant `struct` aggregate. Previously a `todo!()`; now handled as an
// aggregate of field constants (the backend builds it with const_struct).
// This is the single most common csmith construct.

struct S {
    int a;
    int b;
    char c;
    int arr[3];
};

static struct S g = {10, 20, 'A', {1, 2, 3}};

int main(void) {
    return g.a + g.b + g.c + g.arr[0] + g.arr[1] + g.arr[2]; // 10+20+65+1+2+3 = 101
}
