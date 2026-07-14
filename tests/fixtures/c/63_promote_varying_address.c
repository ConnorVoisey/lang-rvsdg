// The read-modify-write address VARIES with the induction variable
// (a[i & 1] alternates between two cells): there is no single cell to
// carry across the loop, and collapsing the two would merge independent
// sums.

int a[2];

int main(void) {
    for (int i = 0; i < 9; i++)
        a[i & 1] += i + 1;
    return a[0] * 10 + a[1];
}
