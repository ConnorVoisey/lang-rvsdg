#!/bin/bash
# llvm-reduce interestingness test: preserve the rvsdg-vs-clang checksum
# MISMATCH (a silent value miscompile where both binaries exit 0 but compute
# different checksums), NOT a crash and NOT undefined behaviour.
#
# csmith generates UB-free, deterministic programs, but llvm-reduce does not
# preserve that -- it freely introduces UB (e.g. truncating main's signature so
# the checksum reads ASLR-randomized pointer bits). Such a program prints a
# different checksum every run, which would trivially "mismatch" clang without
# being a real bug. So each binary is run several times and must be
# deterministic before its output is compared.
LL="$1"

# Per-invocation scratch dir. Under `-j` many copies of this script run at
# once; fixed output names (out_rvsdg/out_clang) would race and clobber each
# other, corrupting the reduction. mktemp gives each chunk its own files.
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
r_bin="$tmp/r"
c_bin="$tmp/c"

# Number of times each binary is run to check for run-to-run determinism.
RUNS="${RUNS:-5}"

# Run "$1" RUNS times; echo its output only if every run exited 0 and produced
# identical output, otherwise return non-zero (crash, timeout, or nondeterminism).
run_stable() {
    local bin="$1" first out i
    first=$( ( timeout 3s "$bin" ) 2>&1 ) || return 1
    for ((i = 1; i < RUNS; i++)); do
        out=$( ( timeout 3s "$bin" ) 2>&1 ) || return 1
        [ "$out" = "$first" ] || return 1
    done
    printf '%s' "$first"
}

# rvsdg build + run. Wrapping the compile in a subshell whose stderr is
# redirected swallows the shell's own "Segmentation fault" report when our
# compiler crashes on an intermediate reduction -- that's a rejected
# candidate, not a result, so it's pure noise during the reduce.
( ./target/release/lang-rvsdg "$LL" -o "$r_bin" -q ) 2>/dev/null || exit 1
r=$(run_stable "$r_bin") || exit 1

# clang reference build + run.
( clang-19 -O0 -w "$LL" -o "$c_bin" ) 2>/dev/null || exit 1
c=$(run_stable "$c_bin") || exit 1

# Both must actually print a checksum (guards against empty-output "mismatch").
echo "$r" | grep -q "checksum = " || exit 1

# The bug: the two compilers disagree on the checksum.
[ "$r" != "$c" ] || exit 1
