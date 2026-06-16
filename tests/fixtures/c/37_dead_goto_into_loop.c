// Regression for the seed-7000 class: an *unreachable* block whose edge points
// into a loop fabricates a phantom second entry vertex, making a perfectly
// reducible loop look irreducible.
//
// `clang -disable-llvm-passes` (the fixture frontend) runs no CFG cleanup, so it
// leaves the `dead` block in place. `dead`'s `goto body` is a real CFG edge into
// the loop body, so the loop ends up with two entry vertices (the header and
// `body`) -- the analysis then tries to build an irreducible-loop entry region
// and chokes. Since `dead` is unreachable it can never execute, so dropping its
// arcs is sound; the loop is then a normal single-entry loop summing 0..4 = 10.
int main(void) {
    int i = 0, s = 0;
    goto start;
dead:               // unreachable: nothing ever branches here
    goto body;      // ...but its edge jumps into the loop body
start:
    while (i < 5) {
body:
        s += i;
        i++;
    }
    return s;
}
