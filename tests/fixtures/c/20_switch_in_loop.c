// Tier 20 — switch inside a loop body. §4.1 (theta body walker).
//
// The theta body walker only handles `Br` / `CondBr` terminators inside
// the loop; a `Switch` inside the body is rejected
// (src/llvm_parser/region/loops.rs, `lower_body_walk`). Supporting it
// means the body walker building an n-arm gamma the same way the acyclic
// path does, with each case arm continuing the walk.

int main(void) {
    int s = 0;
    for (int i = 0; i < 6; i++) {
        switch (i % 3) {
            case 0:
                s += 1;
                break;
            case 1:
                s += 2;
                break;
            default:
                s += 3;
                break;
        }
    }
    return s; // i%3 over 0..5 = 0,1,2,0,1,2 -> 1+2+3+1+2+3 = 12
}
