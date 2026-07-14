// Constant-address expressions: clang's constant folder emits these as
// getelementptr CONSTANT expressions, and under opaque pointers their
// source element type exists only on the expression itself -- it is not
// derivable from the base pointer's type or the index count. The shapes
// here cover element-typed pointer arithmetic past an element, a
// descent into a 2D aggregate, and a pointer-to-pointer cell; the exit
// code pins every address computation.

long arr[8];
long grid[4][4];
long *p = &arr[2] + 1;
long *q = &grid[1][2];
long **pp = &p;

int main(void) {
    for (int i = 0; i < 8; i++)
        arr[i] = i * 10;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            grid[i][j] = i * 4 + j;
    return (int)(*p + **pp + *q);
}
