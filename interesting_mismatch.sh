#!/bin/bash
# llvm-reduce interestingness test: preserve the rvsdg-vs-clang checksum
# MISMATCH (a silent value miscompile where both binaries exit 0 but compute
# different checksums), NOT a crash. The crash guards below stop the reducer
# from drifting into the easier "rvsdg segfaults / clang is fine" finding.
LL="$1"

# Per-invocation scratch dir. Under `-j` many copies of this script run at
# once; fixed output names (out_rvsdg/out_clang) would race and clobber each
# other, corrupting the reduction. mktemp gives each chunk its own files.
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
r_bin="$tmp/r"
c_bin="$tmp/c"

# rvsdg build + run. Wrapping the compile in a subshell whose stderr is
# redirected swallows the shell's own "Segmentation fault" report when our
# compiler crashes on an intermediate reduction -- that's a rejected
# candidate, not a result, so it's pure noise during the reduce.
( ./target/release/lang-rvsdg "$LL" -o "$r_bin" -q ) 2>/dev/null || exit 1
r=$( ( timeout 3s "$r_bin" ) 2>&1 ); rc=$?

# clang reference build + run.
( clang-19 -O0 -w "$LL" -o "$c_bin" ) 2>/dev/null || exit 1
c=$( ( timeout 3s "$c_bin" ) 2>&1 ); cc=$?

# Both must run cleanly (no crash, no timeout) and actually print a checksum,
# so the reducer can't satisfy the test with a crash or empty output.
[ $rc -eq 0 ] && [ $cc -eq 0 ]    || exit 1
echo "$r" | grep -q "checksum = " || exit 1

# The bug: the two compilers disagree on the checksum.
[ "$r" != "$c" ] || exit 1
