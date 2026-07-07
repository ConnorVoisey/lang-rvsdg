// A function pointer flowing through branch joins and a loop header,
// reduced from SQLite. The value type of a global or function reference is
// a pointer; typing it with the referent's type instead makes the loop
// slot carry a function type, which has no LLVM value representation.

static int add_one(int x) { return x + 1; }
static int add_two(int x) { return x + 2; }

int main(void) {
    int (*f)(int) = add_one;
    int acc = 0;
    for (int i = 0; i < 4; i++) {
        acc = f(acc);
        if (acc & 1) {
            f = add_two;
        } else {
            f = add_one;
        }
    }
    return acc - 7;
}
