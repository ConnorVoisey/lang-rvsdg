// Tier 15 — short-circuit logical AND (&&).
//
// `a > 1 && b < 5` lowers (without simplifycfg) to a CFG diamond: the
// right operand `b < 5` is only evaluated when `a > 1` is true. The join
// block carries a phi with TWO predecessors — the condition block itself
// (the short-circuit path, contributing `false`) and the rhs-eval block
// (contributing `b < 5`).
//
// Gamma construction currently rejects this: the phi has no incoming
// value from the "then" arm because that arm IS the condition block.
// Fails at src/llvm_parser/region/branches.rs:333.
//
// Required for csmith differential testing — csmith emits && / || in
// almost every expression.

int main(void) {
    int a = 3, b = 0;
    int r = (a > 1 && b < 5);
    return r; // expect 1
}
