// Tier 28 — switch inside a loop whose cases break / continue / return.
//
// Combines switch-in-loop with a return out of one case, so the loop has
// exits going to both the after-loop block and the function exit, reached
// through the in-body switch dispatch.

int f(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        switch (i % 4) {
            case 0:
                s += 1;
                break; // out of switch, continue loop
            case 1:
                continue; // next iteration
            case 2:
                return s + 50; // out of the function
            default:
                s += 8;
        }
        s += 100;
    }
    return s;
}

int main(void) {
    return f(6); // i=0: s=1+100=101; i=1: continue; i=2: return 101+50=151
}
