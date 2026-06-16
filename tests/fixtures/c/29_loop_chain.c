// Tier 29 — two loops in sequence: the first loop's exit target is the
// second loop's entry (preheader).
//
// Exercises a loop whose exit resumes directly at another loop, with the
// loop-closed value of the first feeding the second.

int main(void) {
    int s = 0;
    for (int i = 0; i < 5; i++) {
        s += i; // 0+1+2+3+4 = 10
    }
    for (int j = 0; j < 3; j++) {
        s += 10; // +30
    }
    return s; // 40
}
