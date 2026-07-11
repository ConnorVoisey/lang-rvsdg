// Memory-effect function attributes: __attribute__((const)) functions are
// memory(none), __attribute__((pure)) are memory(read). LLVM 16 folded the
// old readnone/readonly function attributes into the composite memory(...)
// attribute, which needs its own re-emission.

__attribute__((const)) static int square(int x) { return x * x; }

static int table[4] = {1, 2, 3, 4};

__attribute__((pure)) static int lookup(int i) { return table[i & 3]; }

int main(void) { return square(3) + lookup(2) - 12; }
