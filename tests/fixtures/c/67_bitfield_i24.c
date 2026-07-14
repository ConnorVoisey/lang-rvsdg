// A union whose only member is a 22-bit bitfield: bitfield storage
// units round up to whole bytes, so the union's storage type is a
// 24-bit integer -- a width with no power-of-two name. The type, its
// zero-initialised global, and the masked read-modify-write accesses
// all carry that width. Signed field, so the reads exercise the
// sign-extension path too.

union tight {
    int wide : 22;
};

union tight u;

int main(void) {
    u.wide = -1234;
    u.wide += 34;
    return (u.wide == -1200) ? 42 : 7;
}
