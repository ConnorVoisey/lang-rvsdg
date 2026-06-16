// Tier 17 — `break` taken BEFORE the rest of the loop body runs.
//
// Unlike tier 05 (where the break follows the body's accumulation), here
// the break is the first statement in the body. The loop-closed phi at
// the break exit therefore references the header phi destination at a
// NON-natural exit point — before any body work has executed.
//
// Phase 2 only permits that binding at the natural exit; the general case
// needs the not-yet-implemented demand-analysis pass (paper section 4.x).
// Fails in src/llvm_parser/region/loops.rs (loop-closed phi binding).
//
// csmith emits `break` in arbitrary body positions, so this pattern is
// required for differential testing.

int main(void) {
    int s = 0;
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            break;
        }
        s += i;
    }
    return s; // expect 0+1+2+3+4 = 10
}
