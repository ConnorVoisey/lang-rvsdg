// Accumulation loop with a data-dependent break (a multi-exit loop
// before restructuring): the cell's value at the moment of the early
// exit is what the program observes.

int a[32];

int main(void) {
    for (int i = 0; i < 32; i++)
        a[i] = i;
    int cell[1] = {0};
    for (int i = 0; i < 32; i++) {
        cell[0] += a[i];
        if (cell[0] > 50)
            break;
    }
    return cell[0];
}
