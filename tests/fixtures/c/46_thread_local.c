// A thread-local global, reduced from a coroutine library's per-thread
// scheduler pointer. Accesses go through the llvm.threadlocal.address
// intrinsic, whose operand LLVM requires to actually be thread-local --
// the flag must survive re-emission of the global.

static _Thread_local int counter;

int main(void) {
    counter = 5;
    counter += 2;
    return counter - 7;
}
