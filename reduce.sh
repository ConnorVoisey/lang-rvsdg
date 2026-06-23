#!/bin/bash
# Run llvm-reduce on $1 against the rvsdg-vs-clang mismatch test.
#
# Excludes the `ir-passes` and `simplify-instructions` delta passes: both run
# LLVM's InstCombine, which aborts the whole reduction on this assertion-enabled
# LLVM 19 build with "Instruction Combining did not reach a fixpoint". Every
# other IR delta pass is listed explicitly (llvm-reduce only takes an include
# list). The MIR-only register-* passes don't apply to LLVM IR and are omitted.
#
# Usage: ./reduce.sh <input.ll>
#   The reduced module is written to reduced.ll (llvm-reduce's default output).
set -euo pipefail

INPUT="${1:?usage: ./reduce.sh <input.ll>}"
TEST="${TEST:-interesting_mismatch.sh}"
JOBS="${JOBS:-$(nproc)}"
MAX_ITERS="${MAX_ITERS:-10}"
LOCKFILE="${LOCKFILE:-.reduce.lock}"

# Only one reduce at a time: llvm-reduce writes reduced.ll, so two concurrent
# runs would clobber each other's output. Hold an exclusive lock for the whole
# script (fd 9 stays open until exit) and fail fast if another run holds it.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "error: another reduce is already running (holding $LOCKFILE)" >&2
    exit 1
fi

DELTA_PASSES=(
    strip-debug-info
    functions
    function-bodies
    special-globals
    aliases
    ifuncs
    simplify-conditionals-true
    simplify-conditionals-false
    invokes
    unreachable-basic-blocks
    basic-blocks
    simplify-cfg
    function-data
    global-values
    global-objects
    global-initializers
    global-variables
    di-metadata
    dbg-records
    metadata
    named-metadata
    arguments
    instructions
    operands-to-args
    operand-bundles
    attributes
)
# Join the array with commas for --delta-passes.
passes=$(IFS=,; echo "${DELTA_PASSES[*]}")

# Verify the input is interesting before spending hours on it -- a non-zero
# exit here means the test script doesn't flag the input, so the reduce would
# refuse to start anyway.
if ! ./"$TEST" "$INPUT" >/dev/null 2>&1; then
    echo "error: '$INPUT' is not interesting under $TEST (nothing to reduce)" >&2
    exit 1
fi

llvm-reduce-19 \
    --test "$TEST" \
    --delta-passes="$passes" \
    --max-pass-iterations="$MAX_ITERS" \
    -j "$JOBS" \
    "$INPUT"
