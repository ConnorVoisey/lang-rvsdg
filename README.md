# lang-rvsdg

A RVSDG (Regionalized Value State Dependence Graph) compiler middle-end written in Rust. The goal is to compete with LLVM's optimization quality while being architecturally simpler and achieving significantly faster compile times.

## What is RVSDG?

RVSDG is a graph-based intermediate representation where:

- **All control flow is structured** — no CFG, no basic blocks, no phi nodes in the traditional sense. Instead, control flow is represented by hierarchical structural nodes:
  - **Gamma nodes** — conditionals (if/else), with a condition input and two sub-regions
  - **Theta nodes** — tail-controlled loops, with loop variables threaded through a body region
  - **Lambda nodes** — functions, with a body region
  - **Phi nodes** — mutual recursion (not SSA phi)
- **Data dependencies are explicit edges** — values flow along edges between nodes, making def-use chains trivially available
- **Side effects are explicit via state edges** — memory operations are ordered by threading state tokens, making alias analysis and reordering opportunities visible in the graph structure itself
- **Regions nest hierarchically** — every structural node contains one or more regions, which in turn contain nodes. This gives natural scope boundaries for optimization

The key insight is that many optimizations that require complex analysis on a CFG (dead code elimination, common subexpression elimination, loop-invariant code motion) become structurally obvious or significantly simpler on an RVSDG because the graph encodes the information that CFG-based IRs must recompute.

For the foundational papers, see:
- Bahmann et al., *"Perfect Reconstructability of Control Flow from Demand Dependence Graphs"* (2015)
- Reissmann et al., *"RVSDG: An Intermediate Representation for Optimizing Compilers"* (2020)

## Architecture

### Pipeline

```
C source
  │
  ▼
clang -O1 -disable-llvm-passes    (frontend only, no LLVM opts)
  │
  ▼
opt -passes=mem2reg                (promote allocas to SSA — required for clean IR)
  │
  ▼
LLVM bitcode (.bc)
  │
  ▼
┌─────────────────────────────┐
│  lang-rvsdg                 │
│                             │
│  LLVM IR → RVSDG (parse)    │
│  Optimization passes        │
│  RVSDG → LLVM IR (emit)     │
└─────────────────────────────┘
  │
  ▼
LLVM bitcode (.bc)
  │
  ▼
llc / clang                        (codegen to native)
```

This is the same approach used by the [JLM compiler](https://github.com/phate/jlm). Consuming and emitting LLVM IR means we can:

1. **Benchmark against real-world code** — any C/C++ project that compiles with clang can be fed through the pipeline
2. **Use existing benchmark suites** — SPEC, Embench, compiler benchmarks all work out of the box
3. **Compare optimization quality directly** — run the same code through `clang -O2` vs `clang -O0 | lang-rvsdg | llc` and diff the output
4. **Swap codegen backends** — the plan is to also target Cranelift as an output layer, enabling a comparison of LLVM vs Cranelift codegen quality when lang-rvsdg is doing all the optimization work

### Compile time measurement

The LLVM IR round-trip (serialize → deserialize) is inherently slow and will dominate early compile-time numbers. To get meaningful measurements of actual optimization time, we use chrome tracing (`chrome://tracing`) to instrument passes and isolate the time spent within lang-rvsdg's own work from the LLVM IR I/O overhead.

## Current status

**Prototyping.** The core data structures exist but the LLVM IR parser does not yet construct RVSDG nodes — it iterates the LLVM module and prints debug output. No optimization passes are implemented. No RVSDG-to-LLVM emission exists.

### What exists

- Core RVSDG data structures: nodes, regions, values, state edges
- Type system with arena-based interning and deduplication (scalars, pointers, arrays, functions, structs)
- Builder API for constructing RVSDG graphs (partial — gamma/theta scaffolding)
- LLVM bitcode ingestion pipeline (clang → opt → `llvm-ir` crate)
- LLVM IR parser skeleton

### What's next

- Complete the LLVM IR → RVSDG construction
- RVSDG → LLVM IR emission
- First optimization passes (dead node elimination, constant folding)
- Tracing instrumentation
- Test infrastructure

## Building

Requires:
- Rust (edition 2024)
- clang-17 and opt-17 on `$PATH`
- LLVM 17 development headers (for the `llvm-ir` crate)

```bash
cargo build
cargo run  # runs on examples/c/basic.c
```

## References

- [JLM](https://github.com/phate/jlm) — RVSDG-based compiler that inspired this project's approach
- [RVSDG paper](https://arxiv.org/abs/1912.05036) — Reissmann et al., 2020
- [Cranelift](https://cranelift.dev/) — planned alternative codegen backend
