// A plain global's address (not a function's) flowing through branch joins
// and a loop header. The value type of a global reference is a pointer;
// typing it with the referent's type instead would give the loop slot the
// global's own type (int here) and miscompile the pointer swaps below.

static int a = 3;
static int b = 4;

int main(void) {
    int *p = &a;
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += *p;
        if (*p == 3) {
            p = &b;
        } else {
            p = &a;
        }
    }
    return sum - 14;
}
