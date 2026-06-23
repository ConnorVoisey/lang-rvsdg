#!/bin/bash
LL="$1"
./target/release/lang-rvsdg "$LL" -o out_rvsdg -q 2>/dev/null || exit 1
r=$(timeout 1s ./out_rvsdg 2>&1); rc=$?
# clang reference build + run
clang-19 -O0 -w "$LL" -o out_clang 2>/dev/null || exit 1
c=$(timeout 1s ./out_clang 2>&1); cc=$?
# both must run cleanly and actually print a checksum (no crash/empty drift)
[ $rc -eq 0 ] && [ $cc -eq 0 ]                      || exit 1
echo "$r" | grep -q "checksum = " || exit 1
# the bug: they disagree
[ "$r" != "$c" ]                                    || exit 1
