// Tier 2 -- acyclic branch with mixed continuation points, no loop.
//
// A short-circuit condition `(a && b) || c` guarding two distinct returns
// produces a branch with three sequential tests and TWO shared outcome blocks:
//
//     test a -- false --> test c
//        | true                |
//     test b -- false --> test c
//        | true                | true / false
//     return X            return X / return Y
//
// The two return blocks are each reached from more than one branch, so the
// region has two continuation points rather than one. This is the section 4.2
// branch-restructuring "multiple continuation points" case: a p-demux at the
// top level of the function body (outside any loop body, so not handled by
// LoopBodyExit::Demux). Single-term short-circuits (fixtures 15, 16) reconverge
// at one point and need no demux; this one does.
// Tests: top-level mixed-continuation gamma demux construction (SeqExit::Demux).

int classify(int a, int b, int c) {
    if ((a && b) || c) {
        return 11;
    }
    return 7;
}

int main() {
    int sum = 0;
    sum += classify(1, 1, 0); // a && b -> 11
    sum += classify(0, 0, 1); // c      -> 11
    sum += classify(1, 0, 0); // !ab !c -> 7
    sum += classify(0, 1, 0); // !ab !c -> 7
    return sum;               // 11 + 11 + 7 + 7 = 36
}
