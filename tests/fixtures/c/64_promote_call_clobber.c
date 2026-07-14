// The loop body calls a function that reads AND writes the accumulator
// cell between updates: the cell's intermediate memory values are
// semantically visible to the callee, so its load/store traffic cannot
// be elided across the call (52 with the callee seeing each update,
// different if it sees stale values).

int cell[1];

void bump(void) {
    cell[0] *= 2;
}

int main(void) {
    for (int i = 1; i <= 4; i++) {
        cell[0] += i;
        bump();
    }
    return cell[0];
}
