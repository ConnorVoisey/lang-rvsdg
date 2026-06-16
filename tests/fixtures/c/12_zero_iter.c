// Probe: test-first loop whose condition is false on entry, so the body
// must never execute. Tests the gating gamma's false-arm passthrough:
// the theta exits on iteration zero, projection equals the initial
// loop variable value.

int main() {
    int i = 10;
    while (i < 5) {
        i = i + 1;
    }
    return i; // expected: 10 (body never ran)
}
