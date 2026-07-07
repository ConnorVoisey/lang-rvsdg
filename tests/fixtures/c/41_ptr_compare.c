// Pointer comparisons, reduced from SQLite. LLVM's icmp instruction accepts
// pointer operands (`icmp eq ptr %p, null`), so the compare lowering must
// not assume integer operands.

static int value = 7;

int main(void) {
    int *p = &value;
    int *q = 0;
    if (p == q) {
        return 1;
    }
    if (p != &value) {
        return 2;
    }
    return *p - 7;
}
