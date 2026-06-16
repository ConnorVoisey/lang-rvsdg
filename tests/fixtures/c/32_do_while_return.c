// Tier 32 — do-while loop with an early return inside.
//
// Test-first vs tail-controlled interaction: the do-while body has a
// return exit alongside the natural back-edge / exit.

int f(int n) {
    int s = 0;
    int i = 0;
    do {
        if (s > 6) return s * 10;
        s += i;
        i++;
    } while (i < n);
    return s;
}

int main(void) {
    return f(10); // s: 0,0,1,3,6,10>6 -> return 100
}
