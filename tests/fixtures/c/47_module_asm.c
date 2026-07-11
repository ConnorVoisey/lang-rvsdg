// Module-level inline assembly defining a real symbol (the shape coroutine
// libraries use for hand-written context-switch routines). The assembly
// block must be preserved verbatim through re-emission or the symbol
// vanishes and the link fails.

__asm__(
    ".text\n"
    ".globl asm_five\n"
    "asm_five:\n"
    "  movl $5, %eax\n"
    "  ret\n");

extern int asm_five(void);

int main(void) { return asm_five() - 5; }
