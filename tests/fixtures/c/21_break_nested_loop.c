// Tier 21 — break out of a NESTED loop. §4.1.
//
// The `goto done` leaves the inner loop AND the outer loop at once: the
// inner loop's exit target lies outside the outer loop body. The inner
// loop's exit dispatch must also write the OUTER theta's `q` slot.
//
// Construction now succeeds (the switch-in-body dispatch this produces is
// handled since tier 20), but the result lowers to invalid LLVM
// ("Branch condition is not 'i1' type" at module verification): the nested
// exit produces a gamma whose i32 selector is mis-routed. The real
// break-out-of-nested handling — threading the inner exit through the
// outer theta's `q` — is still missing (src/llvm_parser/region/loops.rs).

int main(void) {
    int s = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i + j == 4) goto done;
            s += 1;
        }
    }
done:
    return s; // breaks at i=0,j=4 after 4 increments -> 4
}
