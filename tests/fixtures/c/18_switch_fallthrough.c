// Tier 18 — switch with fall-through (shared continuation tail). §4.2.
//
// `case 0` falls through into `case 1`'s body, so the case-1 block is a
// continuation reachable from BOTH the switch dispatch (the case-1 label)
// and the end of case 0 — dominated by no single arm. Handled by the
// path-aware walk: each arm lowers the shared block independently,
// resolving its phis for the path that arrived, and binds the join phi
// from the predecessor it exited through (`lower_region` interior-phi
// resolution + `arm_phi_contributions`, src/llvm_parser/region/).

int f(int x) {
    int r = 0;
    switch (x) {
        case 0:
            r += 1;
            /* fall through */
        case 1:
            r += 2;
            break;
        default:
            r += 4;
    }
    return r;
}

int main(void) {
    return f(0); // 0: 1 + 2 = 3
}
