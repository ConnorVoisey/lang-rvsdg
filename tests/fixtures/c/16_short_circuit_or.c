// Tier 16 — short-circuit logical OR (||).
//
// Mirror of tier 15 for `||`: `a > 1 || b < 5` only evaluates the right
// operand when `a > 1` is false. Same join-phi-with-condition-block
// predecessor shape that gamma construction does not yet handle.
// Fails at src/llvm_parser/region/branches.rs:333.
//
// Required for csmith differential testing.

int main(void) {
    int a = 0, b = 0;
    int r = (a > 1 || b < 5);
    return r; // expect 1
}
