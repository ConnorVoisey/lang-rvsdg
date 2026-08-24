# lang-rvsdg

An optimising compiler middle-end built on the RVSDG (Regionalized Value
State Dependence Graph) intermediate representation. Written in Rust, around
24k lines, solo project. It consumes LLVM IR produced by clang and emits
LLVM IR back out, which means every stage gets validated and benchmarked
against real C codebases rather than toy inputs.

The goal is runtime performance near LLVM -O2 at a fraction of the compile
time, with a simpler architecture than LLVM's. The construction, testing and
measurement infrastructure is done and solid; the optimisation passes that
will close the runtime gap are the current work. The status section below
gives an accurate picture of both halves.

## What works today

The full pipeline runs on real code. The SQLite amalgamation (roughly 250k
lines of C), Lua, and all 30 PolyBench kernels compile end to end: LLVM IR
in, RVSDG construction, optimisation, LLVM IR out, native binary. The
resulting SQLite shell and Lua interpreter produce output identical to
clang-built references on smoke workloads covering a broad SQL feature sweep
and Lua language semantics.

Construction and control-flow reconstruction follow Bahmann, Reissmann,
Jahre and Meyer, "Perfect Reconstructability of Control Flow from Demand
Dependence Graphs" (2015). That paper's approach, predicate continuation
form with gamma/theta restructuring, handles arbitrary control flow without
imposing structure on the generated code.

Correctness is tested through fuzzing, unit and integration tests. A differential
tester compiles each input with this compiler and with clang, runs both
binaries and diffs their behaviour. In fuzzing mode it generates programs
with Csmith; failing cases get minimised with llvm-reduce. This has caught,
and led to fixes for, a long list of real miscompilations. The most recent
run of 10,000 generated programs found zero behavioural mismatches and one
compiler crash on an unsupported construct. Alongside that sit a
structural graph verifier (IDs, ownership, scoping, typing, state edges,
predicate form) and around 300 unit tests.

Performance work is measured, not guessed at. A purpose-built compile-time
harness uses Cachegrind instruction counts as the primary signal, because
they are deterministic and do not drown regressions in wall-clock noise,
plus per-phase breakdowns and HTML regression reports against saved
baselines. A separate harness runs PolyBench for runtime and compile-time
comparisons against clang -O2.

## Current numbers, with their caveats

From the PolyBench suite, large dataset, all 30 kernels passing:

- Runtime: median 1.9x slower than clang -O2, ranging from parity to 3.7x.
  This is expected. The mid-end currently performs only dead node
  elimination, so the comparison is roughly mem2reg plus codegen against
  LLVM's full -O2 pipeline. Closing this gap is what the project is for.
  Branch-heavy binaries like Lua and SQLite are also slower than clang's
  builds; a known contributor is that the RVSDG destruction step, which
  rebuilds control flow for LLVM emission, is currently naive.
- Compile time, IR to object, both compilers given the same input bitcode:
  median 0.86x of clang -O2, i.e. currently faster, though partly because
  less optimisation work is being done. How much of that headroom survives
  as real passes land is exactly the question the benchmark harness exists
  to answer.

## Why RVSDG

RVSDG is a graph IR in which all control flow is structured. There is no
CFG and there are no basic blocks. Instead:

- Gamma nodes represent conditionals, with a predicate input and one
  sub-region per branch.
- Theta nodes represent tail-controlled loops, with loop variables threaded
  through a body region.
- Lambda nodes represent functions.
- Data dependencies are explicit edges, so def-use chains come for free.
- Side effects are ordered by explicit state edges. Memory operations
  thread state tokens, which makes aliasing and reordering opportunities
  visible in the graph itself.
- Regions nest, giving natural scope boundaries for optimisation.

The payoff is that optimisations needing substantial analysis on a CFG,
such as dead code elimination, common subexpression elimination and
loop-invariant code motion, become structurally simple, because the graph
already encodes what CFG-based IRs have to recompute.

Foundational papers: Bahmann et al. 2015 (construction and destruction)
and Reissmann et al. 2020 (the RVSDG IR itself). Links at the bottom.

## Architecture

```
C source
   |
   v
clang -O1 -Xclang -disable-llvm-passes    frontend only, no LLVM optimisation
   |
   v
opt -passes=mem2reg                       promote locals to SSA
   |
   v
LLVM bitcode (.bc)
   |
   v
+------------------------------------+
|  lang-rvsdg                        |
|                                    |
|  parse LLVM IR, construct RVSDG    |
|  (control-flow restructuring)      |
|  optimisation passes               |
|  graph verification                |
|  emit LLVM IR                      |
+------------------------------------+
   |
   v
LLVM bitcode (.bc)
   |
   v
clang / llc                               codegen to native
```

This is the same approach the [JLM compiler](https://github.com/phate/jlm)
takes. Only mem2reg runs before construction. LLVM's restructuring and
optimisation passes are deliberately disabled: restructuring control flow
is the RVSDG construction's own job, and mid-level optimisation is what
this project is here to do, so letting LLVM do either would bias every
later benchmark.

## Status

As of August 2026.

Working:

- LLVM IR parsing and RVSDG construction, including globals, function
  attributes, and the restructuring of arbitrary control flow into
  gamma/theta form per the 2015 paper
- RVSDG to LLVM IR emission and native codegen via clang/llc
- Optimisation passes: dead node elimination, plus state-edge rerouting
  around constructs that merely pass state through (which is what lets the
  liveness pass collect pure constructs)
- Structural verifier: ID validity, node ownership, scoping, typing, state
  edge discipline, predicate form
- Differential tester and Csmith fuzzer, compile-time benchmark harness,
  PolyBench runtime harness, chrome-tracing instrumentation, heap profiling

In progress:

- Memory alias analysis. Pointer provenance tracking and escape analysis
  run at construction time already; splitting the single memory state chain
  into per-alias-class chains, so independent memory operations stop being
  artificially ordered, is designed and being built.

Planned:

- The passes that will actually close the runtime gap: common subexpression
  elimination, loop-invariant code motion, scalar promotion, induction
  variable analysis
- Smarter RVSDG destruction; the current control-flow reconstruction is
  naive, which costs runtime performance on branch-heavy code
- Function summaries for interprocedural precision
- A non-LLVM codegen backend (Cranelift, potentially custom) for comparing backends while this
  project does all the optimisation

## Correctness testing

The differential tester compiles a C file with both this compiler and
clang, runs both binaries, and compares stdout and exit codes. With
`--count N` it fuzzes instead: Csmith generates programs and the same
comparison runs in a loop, in parallel, with every subprocess under a
timeout, so a compiler crash, an abort() or an infinite loop becomes a
reported finding rather than a stuck driver. Failing inputs are saved to
`difftest-findings/`.

```sh
# test specific files
cargo run --release --bin difftest -- some.c

# fuzz 1000 csmith programs
cargo run --release --bin difftest -- --count 1000
```

A found miscompilation is then shrunk with llvm-reduce while preserving the
behavioural mismatch:

```sh
clang -I /usr/include/csmith-2.3.0 -O1 -Xclang -disable-llvm-passes \
  -emit-llvm -c difftest-findings/7015.c -o difftest-findings/7015.bc
opt -passes=mem2reg -S difftest-findings/7015.bc -o difftest-findings/7015.ll
llvm-reduce --test interesting.sh difftest-findings/7015.ll -j $(nproc)
```

## Benchmarking

`compile_bench` measures compile time with Cachegrind instruction counts as
the primary signal, so regressions are not lost in wall-clock noise, plus
optional wall/RSS/phase-breakdown passes, and renders an HTML report
comparing runs against saved baselines. Both compilers are timed on the same
pre-staged bitcode, so neither is charged for the shared clang frontend.

```sh
cargo run --release --bin compile_bench -- --polybench path/to/polybench
```

Results are recorded under `bench-results/` with the git SHA and machine
metadata, so numbers are reproducible and comparable across commits.

## Building

Requires:

- Rust (edition 2024)
- LLVM 22: `clang` and `opt` on `$PATH`, plus the LLVM 22 development
  libraries (for inkwell and the llvm-ir crate)

```bash
cargo build --release

# if llvm-config isn't the system default, point the llvm-sys build at it:
LLVM_SYS_221_PREFIX=$(llvm-config --prefix) cargo build --release

# run the test suite
cargo test
```

## References

- [JLM](https://github.com/phate/jlm), the RVSDG-based compiler whose
  LLVM-IR-round-trip approach this project shares
- [Bahmann et al. 2015](https://dl.acm.org/doi/10.1145/2693261), the
  construction and destruction algorithms this project implements
- [Reissmann et al. 2020](https://arxiv.org/abs/1912.05036), the RVSDG IR
- [Cranelift](https://cranelift.dev/), the planned alternative codegen
  backend
