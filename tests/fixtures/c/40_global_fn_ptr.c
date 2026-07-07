// A global struct holding a function pointer, reduced from SQLite's
// sqlite3_pcache_methods2 registration table. In LLVM IR every global
// precedes every function, so the initializer's FuncAddr constant references
// a function the lowering has not reached yet: function declarations must be
// registered before global initializers are lowered.

struct ops {
    int tag;
    int (*init)(int);
    void *unused;
};

static int init_impl(int x) { return x + 41; }

struct ops global_ops = {7, init_impl, 0};

int main(void) { return global_ops.init(1) - global_ops.tag; }
