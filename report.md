# RVSDG Construction Report

A walkthrough of what the LLVM-IR-to-RVSDG construction in this
codebase currently does, mapped against the paper Bahmann, Reissmann,
Jahre, Meyer (2015) "Perfect Reconstructability of Control Flow from
Demand Dependence Graphs" (ACM TACO 11(4), Article 66). The paper
text is reproduced verbatim at `perfect_reconstruction.txt` in the
repository root.

This report is written so you can follow it without having read the
paper first. Every paper term gets a plain-English gloss in
parentheses the first time it appears.

## 0. A short vocabulary primer

Before anything else, the words you need to keep straight.

- **LLVM IR** - the input we receive. Looks like assembly: a list of
  named "basic blocks", each ending in a branch or return.
- **Basic block** - a straight run of instructions with no branches
  in the middle. Branches happen only at the end.
- **CFG (Control Flow Graph)** - the graph whose nodes are basic
  blocks and whose edges are the branches between them.
- **SSA (Static Single Assignment)** - the property that each
  variable is assigned exactly once. LLVM IR is in SSA form.
- **Phi node** - the SSA construct at the top of a basic block that
  says "the value of this variable depends on which predecessor
  block we arrived from". Required because in SSA you cannot
  reassign, so merging two paths needs a phi to pick the right
  incoming value.
- **RVSDG (Regionalized Value State Dependence Graph)** - the
  output we produce. A dataflow graph where nodes are operations
  and edges are value/state dependencies. Control flow is
  represented by two special node kinds:
  - **Gamma node** - an if/else (or n-way switch). Has a predicate
    input and one sub-region per branch arm. Output values are
    picked from whichever arm matched the predicate.
  - **Theta node** - a loop. Has a body sub-region that produces a
    "should we loop again?" boolean each iteration. Loop-carried
    variables flow through "loop_var slots".
- **SCC (Strongly Connected Component)** - a cluster of basic
  blocks that can all reach each other through the CFG. In
  practice this means "a loop and its body". A function with no
  loops has no non-trivial SCCs; nested loops produce nested SCCs.
- **Dominator** - block A dominates block B if every path from the
  function entry to B passes through A. The **immediate dominator
  (idom)** of B is the closest such A.
- **Post-dominator** - block A post-dominates block B if every path
  from B to the function exit passes through A. The **immediate
  post-dominator (ipdom)** of B is the closest such A. This is
  the structural "where do all paths after B reconverge?".
- **LCSSA (Loop-Closed SSA Form)** - a CFG canonicalisation that
  inserts a phi at every loop exit block for every loop-defined
  value used outside the loop. Makes it mechanical to identify
  what a loop "produces".

## 1. The big picture: what we are translating to what

We take LLVM IR (a flat CFG with phi nodes, no syntactic loops or
if/else) and produce an RVSDG (a tree of gamma and theta nodes with
no phi nodes). The challenge is that LLVM IR has no notion of "this
is a loop" or "this is an if/else"; we have to discover the
structure from the CFG and then construct the corresponding RVSDG.

The construction lives under `src/llvm_parser/region/`. The entry
point is `RegionLowerer::lower_region` in
`src/llvm_parser/region/mod.rs`, which walks the function in source
order and dispatches to:

- `branches::lower_cond_branch` and `branches::lower_switch` when
  it sees an if/else or switch (builds a gamma).
- `loops::lower_scc_as_theta` when it sees the entry of an SCC
  (builds a theta).

## 2. Where we sit in the paper

The paper specifies three layered transforms:

1. **Section 4.1 - Loop restructuring.** For every SCC (loop), the
   paper introduces synthetic helper variables and rewrites the
   loop so it has exactly one entry, one back-edge, and one exit.
   This makes the loop body a self-contained graph that can be
   wrapped in a theta node.
2. **Section 4.2 - Branch restructuring.** For acyclic regions
   where if/else arms do not cleanly reconverge (think: `if (x) {
   return; }` mid-function), introduce another helper variable
   that records "where would we have gone?" so the messy CFG can
   be rewritten as a clean tree of gamma nodes.
3. **Section 4, paragraph after Definition 4.1 - BuildRVSDG.**
   Once the CFG is structured (output of steps 1 + 2), a linear
   pass walks it like an interpreter would, mapping each construct
   to the appropriate RVSDG node.

This compiler implements:

- A direct version of step 3 (BuildRVSDG) on the input CFG,
  expecting the input to be **mostly structured already**.
- Step 1 (loop restructuring) **only for single-entry,
  single-back-edge SCCs**. Multi-exit SCCs are handled in the
  same pattern: the helpers `q` and `r` (explained below) are
  emitted at every "leaf" of the loop body, and a gamma after the
  theta routes on `q` to the right exit.
- **No part of step 2 (branch restructuring) for the unstructured
  case.** When `lower_region` meets an if/else whose arms do not
  reconverge at the immediate post-dominator, it panics with a
  message that names the missing transform.

The gap is bridged today by the LLVM optimisation pipeline run on
the input: `sroa,mem2reg,loop-simplify,lcssa`. Of these:

- `sroa,mem2reg` - convert C local variables (which live in
  memory) into SSA registers. These are necessary regardless of
  the algorithm; they stay forever.
- `loop-simplify` - rewrites every loop into the single-entry,
  single-latch, dedicated-exit shape the paper's loop
  restructuring would produce. Stand-in until we implement the
  multi-back-edge transform ourselves.
- `lcssa` - inserts loop-closed phi nodes (defined above). Lets
  the construction read off what a loop produces. Stand-in for
  some of the symbol-table mechanics the paper does inline.

Once the construction handles unstructured branches and
multi-entry / multi-back-edge SCCs natively, the final pipeline
target is `sroa,mem2reg` only.

## 3. What is implemented, mapped to the paper

### 3.1 SCC identification

**Paper says** (section 4.1, paragraph 1): identify all SCCs using
Tarjan's algorithm.

**Code:** `src/llvm_parser/strongly_connected_components.rs` runs
Tarjan on the function's CFG. `src/llvm_parser/scc_tree.rs` then
builds a nesting hierarchy where outer SCCs contain inner SCCs as
sub-trees, pre-computes for each SCC the four arc sets the paper
names (see 3.2), and records `scc_entry_block_to_id` so the walker
can detect inner-SCC entry points in O(1).

### 3.2 Loop arcs

**Paper says** (section 4.1, paragraph 2): for each SCC, identify
four sets of arcs and vertices:

- **Entry arcs (`A^E` in the paper, "the set of entry arcs")** -
  arcs coming from outside the SCC into the SCC.
- **Entry vertices (`v^E_k`)** - the blocks inside the SCC that
  entry arcs land on. For a normal loop there is exactly one
  (the loop header).
- **Exit arcs (`A^X`)** - arcs going from inside the SCC to outside.
- **Exit vertices (`v^X_k`)** - the blocks outside the SCC that
  exit arcs land on. For a normal loop with no `break` there is
  exactly one (the block after the loop).
- **Repetition arcs (`A^R`)** - arcs from inside the SCC back to
  an entry vertex. These are the back-edges. For a normal loop
  with `loop-simplify` applied there is exactly one (from the
  latch back to the header).

**Code:** stored on `LoopArcs` in `src/llvm_parser/loop_arcs.rs`,
attached to each `SccTreeNode`. The Phase 2 construction reads
`entry_blocks` (paper's entry vertices), `exit_arcs` (paper's exit
arcs), and `repetition_arcs` (paper's back-edges) from there.

### 3.3 Theta construction

**Paper says** (section 4.1, end): for each SCC produce a theta node
whose body is the restructured loop body. The theta has two
helper predicates:

- **`r` (the repetition predicate)** - a boolean computed at the
  bottom of each iteration. `r = 1` means "loop again"; `r = 0`
  means "exit".
- **`q` (the vertex-selector predicate)** - a small integer
  computed alongside `r`. When `r = 1` it says which entry vertex
  to resume at on the next iteration; when `r = 0` it says which
  exit vertex to resume at after the loop.

The paper's restructuring writes assignments `q := k` on entry
arcs, `q, r := k, 1` on repetition arcs, and `q, r := k, 0` on
exit arcs, where `k` is the index of the targeted vertex.

**Code:** `RegionLowerer::lower_scc_as_theta` in
`src/llvm_parser/region/loops.rs:170`. After validating the SCC's
shape (it bails on multi-entry and multi-back-edge, see section 5
below) it:

1. Builds a `LoopVarInits` buffer holding the initial values of
   the theta's loop-carried slots: one slot per header phi, one
   per outer-scope live-in, one per loop-closed-extra slot (see
   3.6).
2. Appends one more slot for `q`, initialised to a placeholder.
3. Calls `RegionBuilder::theta`, which builds the theta region
   and lets us populate its body via a closure.
4. Inside the body closure, runs the body walker (see 3.5).
5. Returns `LoopResult { condition: r, next_state, next_vars }`
   where `r` becomes the theta's repetition predicate (paper's
   `r`) and `next_vars` is the slot-and-`q` tuple flowing into
   the next iteration or out of the theta.

The paper's `q := k`, `q, r := k, 1`, `q, r := k, 0` assignments
are realised by the body walker's leaf emission: every leaf
produces a value tuple `(slot values..., q, r)` that the theta
reads as "what changed this iteration".

### 3.4 Recursive SCC dispatch

**Paper says** (section 4.1, last paragraph before 4.2): after
restructuring the outer SCC, the loop body (referred to as `L*`,
which is the paper's notation for "the inside of the restructured
loop, treated as its own closed graph") is itself a closed CFG
and the algorithm can be applied recursively on it. In practice
this means: nested loops get their own theta nodes, built first.

**Code:** the body walker checks `scc_entry_block_to_id[block]` at
every visited block other than its own loop header. When the
block is the entry vertex of an inner SCC, the walker calls
`lower_scc_as_theta` recursively and then resumes walking at the
inner SCC's exit target (`loops.rs:903`). The SCC tree's nesting
guarantees innermost loops are restructured first.

### 3.5 Body walker - the gamma tree inside the loop

**Paper says** (section 4.1, end): the loop body is itself
processed by the construction (the paper says BuildRVSDG\*, which
is the linear walker plus section 4.2 branch restructuring,
applied to the acyclic body of the SCC).

**Code:** the walker functions in `loops.rs` realise a **subset**
of that recursive processing in-line:

- `lower_body_walk` (`loops.rs:881`) lowers one block at a time.
  Each call: dispatches inner SCCs, resolves interior-join phis
  (see 3.7), lowers non-phi instructions, then dispatches on the
  terminator.
- `walk_target` (`loops.rs:982`) decides whether a branch target
  is the loop header (paper: a repetition arc), outside the SCC
  (paper: an exit arc), or another intra-body block (continue
  walking).
- `lower_body_cond_branch` (`loops.rs:1005`) builds a binary
  gamma at every CondBr inside the body; each arm recursively
  walks its target.
- `make_rep_leaf` / `make_exit_leaf` (`loops.rs:1101` /
  `loops.rs:1125`) emit the leaf tuple corresponding to the
  paper's `q, r := k, 1` and `q, r := k, 0` assignments. `r` is
  an `i1`: 1 for repetition leaves, 0 for exit leaves. `q` is an
  `i32`: the dispatch index of the repetition or exit arc.
- `make_leaf_slot_values` (`loops.rs:1056`) computes the slot
  payload of each leaf. For each header phi slot, it resolves
  the back-edge operand if defined along this path; otherwise it
  falls back to the slot's input. The "slot input" is the header
  phi's destination name, which `lower_scc_as_theta` seeds in
  the symbol table as `body_rb.param(slot_idx)`. This is the
  RVSDG equivalent of "the value of this loop variable at the
  start of this iteration".

The leaf tuple's three parts (slot values, q, r) collapse the
paper's separate "q assignment per arc", "r assignment per arc",
and "loop variable updates" into a single value vector at every
leaf. Each gamma inside the body merges its two arms' tuples
slot-by-slot.

### 3.6 Theta loop_var slots

**Paper says** (section 4, paragraph 4, plus 4.1): the theta's
inputs and outputs carry the loop's "variant values" - the values
that change across iterations. Constant-within-loop references
would be free edges into the body (the paper does not model them
as slots).

**Code:** `analyze_loop` (`loops.rs:546`) classifies three
categories of theta loop_var slots:

1. **Header phi slots** (one per phi at the loop header block).
   Each slot's initial value is the phi's preheader operand
   (its value on entry to the loop). The slot's value AFTER the
   loop exits is the value of that header phi at the iteration
   that took the exit.
2. **Live-in slots** (one per outer-scope SSA name referenced
   inside the body, excluding header phi preheader operands).
   Each slot's initial value is the resolved outer value; the
   slot passes through unchanged on every iteration. This is a
   pragmatic extension - the paper would let body references
   reach into the outer scope directly. Our `RegionBuilder` API
   only exposes parameter values to a sub-region, so we have to
   thread live-ins as loop_var slots.
3. **Loop-closed-extra slots** (one per body-internal SSA name
   that some exit's loop-closed phi references; the LCSSA
   "sub-case C" in 3.8). Each iteration writes the body's
   current value into the slot; the post-theta projection reads
   it.

The contiguous buffer is owned by `LoopVarInits` with markers
splitting it into the three regions, plus one extra slot
appended in `lower_scc_as_theta` for `q`.

### 3.7 Interior-join phi resolution

**Paper says** (section 4 + section 4.2): join blocks inside an
acyclic region are handled either by the linear symbolic-execution
walker (when both arms reach the join through the same gamma) or
by branch restructuring (section 4.2) when arms diverge into
different sub-regions.

**Code:** the body walker takes the simpler path. Because each
gamma arm walks the body recursively and independently, two arms
that converge at an interior block J both visit J. The walker
takes a `prev: Option<BasicBlockId>` carrying the predecessor we
arrived from along this arm; on entry to J it resolves every phi
at the top of J by looking up which incoming corresponds to `prev`
and binding the phi destination in this arm's symbol table to
that incoming's value. Subsequent uses in J (or any block the arm
walks past J) resolve normally.

The trade-off: instructions after J that both arms walk through
get lowered once per arm. The RVSDG ends up correct but larger
than what a section-4.2 gamma at J would produce. Folding the
duplicates is a later optimisation; the priority here is
correctness.

### 3.8 Loop-closed phi (LCSSA) bindings

The paper does not name LCSSA - it operates on a CFG without phis
and lets the linear BuildRVSDG\* (section 4 paragraph 4) handle
phi-to-gamma binding. Our input IR is post-LCSSA so loop-closed
phis appear at exit blocks; `analyze_loop` classifies each one
(`loops.rs:617` - `loops.rs:734`):

- **Sub-case A** - the loop-closed phi's incoming operand equals
  some header phi's back-edge operand. Binding:
  `LcssaBinding::HeaderPhi { index }`. The post-theta projection
  of that header phi's slot is the value the loop-closed phi
  wants: the back-edge operand for the iteration that took the
  exit.
- **Sub-case B** - the loop-closed phi's incoming operand equals
  some header phi's destination, **and** the exit arc's source
  is the loop header itself. Same binding as A. The leaf-tuple
  fallback (header phi destination = theta_param) gives the
  right value because at a header-source exit the body has not
  executed this iteration, so the slot's input value (= header
  phi dest) is the correct LCSSA value.
- **Sub-case C** - the loop-closed phi references a body-internal
  SSA value not tied to a header phi. A fresh "lcssa_extra" slot
  is allocated (see 3.6); the walker writes the body-internal
  value into that slot on every iteration; the post-theta
  projection is the LCSSA value.

Sub-case B at a non-header exit source is rejected with a clear
"needs the not-yet-implemented demand-analysis pass" message
(`loops.rs:706`). The paper's symbolic-execution algorithm
handles this implicitly via the symbol-table mechanics, but we
do not have that machinery yet.

### 3.9 Multi-exit post-theta dispatch

**Paper says** (section 4.1, exit handling): exit arcs are
replaced with `q, r := k, 0`; control funnels through `v^T*` (the
paper's name for the synthetic single tail vertex it inserts at
the end of the loop); on exit, `v^X*` (the synthetic single exit
vertex) demultiplexes on `q` to the original exit vertex `v^X_k`.

**Code:** for loops with more than one exit arc,
`lower_scc_as_theta` invokes `lower_exit_dispatch`
(`loops.rs:447`). This:

1. Computes the post-theta join via `compute_exit_targets_join`,
   which finds the lowest common ancestor of the exit targets in
   the post-dominator tree (`loops.rs:387`). If no common
   post-dominator exists, the function bails - that case is
   genuinely paper section 4.2 territory.
2. Computes per-arm live-ins by reusing
   `branches::compute_arm_live_ins` on the union of arm-block
   sets.
3. Builds a gamma whose condition is `q` (paper's exit-vertex
   demultiplexer). For the 2-exit case the gamma lowering's
   binary-branch path requires an `i1`, so the dispatcher
   coerces `q` via `icmp eq q, 0` before passing it in.
4. Each arm walks from its exit target to the join, with its
   loop-closed phi destinations pre-bound to the corresponding
   theta projections.
5. After the gamma, the dispatcher binds join-block phi
   destinations in the outer symbol table and returns the join
   block as the resume point.

## 4. Pragmatic deviations from the paper

These are intentional simplifications, each accompanied by an
in-code rationale comment:

- **`q` indexed by exit arc, not exit vertex.** Two exit arcs to
  the same destination block get distinct `q` values, producing
  a redundant 2-arm dispatch where the paper would use one `q`
  value and one gamma arm. Functionally correct (both arms walk
  to the same target); enlarges the RVSDG.
- **Live-ins as theta loop_var slots.** Discussed in 3.6. Forced
  by the builder API's region-isolation rule.
- **Path-aware phi resolution instead of section-4.2 sub-gammas
  inside the body.** Discussed in 3.7. Each gamma arm re-lowers
  shared blocks once per arm.
- **`q` placeholder on repetition leaves.** The walker emits
  `q = 0` on rep leaves because `q` is read only when `r = 0`
  (loop exit) and the post-theta dispatch is the only consumer.
  For single-entry SCCs this coincides with the paper's `q := 0`
  assignment on the repetition arc.
- **Post-theta join via post-dominator LCA.** The paper would
  derive the join structurally during section-4.2 partitioning.
  Our LCA is equivalent when a common post-dominator exists;
  otherwise we bail.

## 5. Cases the construction currently rejects

Each is gated with a `bail!` that names what needs to be added:

- **Multi-entry SCCs** (`loops.rs:175`). The paper's section 4.1
  describes the entry-side `q := k` demultiplexer; we have not
  built it. Workaround: run the LLVM `fix-irreducible` opt pass
  upstream to fold multi-entry SCCs into single-entry shape.
- **Multi-back-edge SCCs** (`loops.rs:185`). The paper's `v^T*` /
  `v^E*` construction collapses multiple repetition arcs through
  a single tail vertex; we require the input to already have a
  single latch. Workaround: `loop-simplify`.
- **Sub-case-B LCSSA at non-header exits** (`loops.rs:706`).
  Needs the symbol-table-tracking equivalent of the paper's
  BuildRVSDG\*.
- **Break-out-of-nested-loop** (`loops.rs:911`). When an inner
  SCC's exit lands outside the outer SCC's body, the outer
  walker cannot recover which outer-SCC exit arc index to emit.
  Needs the inner SCC's dispatcher to write into the outer
  theta's `q` slot as well.
- **Unstructured branches inside acyclic regions**
  (`region/mod.rs:154`). A CondBr or Switch with no immediate
  post-dominator panics. This is exactly the paper's section 4.2
  scope (the auxiliary continuation predicate `p`, which is the
  third helper variable alongside `q` and `r`; it records "which
  continuation point would we have reached?" when an arm
  short-circuits past the natural join).
- **Switch terminator inside a loop body** (`loops.rs:950`). The
  body walker only knows about Br and CondBr; adding switch is
  mechanical but not done.
- **Early return / unreachable inside a loop body**
  (`loops.rs:956`, `loops.rs:962`). Same reason; section 4.2
  territory in the paper's framing.

## 6. Future steps and how they fit in

The next phases of work all extend the construction either by
adding paper-faithful transforms or by lifting current
restrictions. Each integration point names the relevant code
location so the connection to existing code is unambiguous.

### 6.1 Section 4.2 branch restructuring (auxiliary `p`)

**What:** for acyclic regions whose CondBr/Switch arms do not
reconverge at the immediate post-dominator, insert the auxiliary
continuation predicate `p`: `p := k` along every arm that
short-circuits to a tail continuation point, then `branch p` in
the tail demultiplexes to the original continuation. (`p` is the
paper's third helper variable; semantically it is to branches
what `q` is to loop exits.)

**Where it goes:** the panic at `region/mod.rs:154` (CondBr) and
`region/mod.rs:165` (Switch) becomes a call into a new
`branches::restructure_unstructured_branch` helper. The helper
returns a synthetic acyclic sub-region whose entry is the
original CondBr's predecessor and whose exit is the synthetic
tail demultiplexer; the rest of `lower_region` then proceeds
normally on the structured form.

**Effect on existing code:** none of the structured-branch code
in `branches.rs` changes - it already takes a join block and a
set of arm sub-regions. The restructure step produces exactly
that input from an unstructured one.

**Removes restrictions:** "no immediate post-dominator" panic;
switch fall-through tails; early return inside conditionals.

### 6.2 Multi-back-edge SCC handling

**What:** the paper's `v^T*` / `v^E*` construction. When an SCC
has more than one repetition arc, synthesise a single tail
vertex through which all of them flow, with `q, r := k, 1` per
repetition arc.

**Where it goes:** the bail at `loops.rs:185` becomes a small
preprocessing helper in `loops.rs` that, before
`lower_scc_as_theta` builds the theta, recognises the
multi-back-edge case and folds the repetition arcs into one
logical back-edge whose source is a virtual tail vertex. The
body walker's leaf code already emits `q` and `r`, so the walker
itself does not change; the analysis phase needs to ensure
header phis treat any of the original back-edge sources as the
in-body predecessor.

**Effect on existing code:** `analyze_loop`'s header phi
identification gets a small generalisation (any block in the SCC
body, not just the unique latch). The walker's `walk_target`
already treats `target == walker.header` as the repetition leaf
condition - no change there.

**Removes restrictions:** multi-back-edge SCCs; lets us drop
`loop-simplify` from the opt pipeline.

### 6.3 Multi-entry SCC handling (auxiliary `q` for entries)

**What:** the entry side of the paper's section 4.1 transform.
Replace each entry arc with `q := k`; a single synthetic entry
vertex demultiplexes on `q` to the original entry vertex inside
the body.

**Where it goes:** the bail at `loops.rs:175` becomes a
preprocessing step that, when `entry_blocks.len() > 1`, picks an
order for the entries, rewrites every entry arc to `q := k` going
to a synthetic header, and updates the SCC body's first block to
the synthetic header. The theta's loop_var slot for the synthetic
`q` is fed by the rewritten entry assignments. The body walker
dispatches on `q` immediately at the synthetic header to fan out
to the original entry vertices.

**Effect on existing code:** the body walker grows a new "fan-out
on entry q" step at the top of every iteration; the synthetic
header becomes the loop header from the walker's perspective.
Existing single-entry SCCs hit the same code path with a trivial
1-arm fan-out (or a fast-path).

**Removes restrictions:** multi-entry SCCs; lets us drop
`fix-irreducible` from the opt pipeline once it becomes the only
remaining preprocess.

### 6.4 Demand analysis for general sub-case-B

**What:** the paper's BuildRVSDG\* (section 4 paragraph 4) does a
linear symbolic-execution pass with a symbol table that already
knows, at any point in the body, what value every name resolves
to. With that, sub-case-B at any exit (not just header-sourced
ones) is just "look up the header phi destination in the symbol
table at the exit's source block".

**Where it goes:** the current body walker accumulates a
`name_to_value` map as it walks - this IS the paper's symbol
table for the body. The piece missing is propagating that map's
state across the body's CondBr arms cleanly enough that
`analyze_loop` can ask "what does this LCSSA's incoming resolve
to at this exit?" rather than relying on the leaf-tuple fallback.

Two integration options:
1. Run a precomputation pass that, per exit arc, walks the body
   from the header to the exit's source and records the symbol
   table state at that point. `analyze_loop` then resolves
   sub-case-B by table lookup.
2. Let the walker emit sub-case-B values directly at the exit
   leaf (it already has the right name_to_value at that point).
   This is cleaner but couples `analyze_loop` and the walker
   more tightly.

**Effect on existing code:** option 2 is the lighter touch and
fits the walker's current shape - the leaf-tuple slot computation
in `make_leaf_slot_values` would gain a "this slot's value is
header_phi_dest_at_this_leaf" branch for sub-case-B exits, fed
by the resolved name in the walker's current name_to_value.

**Removes restrictions:** sub-case-B at any exit; many real-world
loops with early returns that carry loop-variables to LCSSA phis.

### 6.5 Break-out-of-nested-loop

**What:** when an inner SCC has an exit arc going outside the
outer SCC, the outer theta's `q` slot must receive a value that
the outer post-theta dispatch can route. The inner theta's exit
dispatch needs to write the outer's `q` AND the outer's `r`
appropriately on its way out.

**Where it goes:** `lower_scc_as_theta` for the inner SCC needs
to know, at construction time, what outer SCCs it sits inside
and which of those outer SCCs each of its exits actually escapes.
The SCC tree already records nesting; we would extend the inner
dispatcher to take a stack of outer (`q`, `r`) writes (one per
outer level the exit escapes) and emit them as the inner leaf
fires.

**Effect on existing code:** the bail at `loops.rs:911` becomes
the entry to this multi-level dispatch construction. The walker's
leaf emission gains a "this exit escapes N outer levels, write
their `q` / `r` slots accordingly" step.

**Removes restrictions:** break-out-of-nested-loop.

### 6.6 Folding duplicated body lowering

**What:** the path-aware phi resolution (see 3.7) lowers
post-join body instructions once per gamma arm. After a structural
section-4.2 pass lands (item 6.1), the body walker can use the
same mechanism to build gammas at interior joins inside the loop
body, folding the duplication.

**Where it goes:** once `branches.rs`'s gamma-build code can run
against an interior sub-region of the body, the walker's
`lower_body_cond_branch` chooses between two strategies based on
the CondBr's structural shape: either build a leaf-tuple gamma
(today's behaviour, for arms terminating at rep/exit leaves) or a
section-4.2 gamma whose join sits at the interior block and whose
output is the join's phi values.

**Effect on existing code:** purely additive - the leaf-tuple
path remains for arms that do not share a body convergence.

**Removes restrictions:** none directly; this is a compile-time
and RVSDG-size optimisation.

### 6.7 Optimisation passes on the RVSDG

Once the input IR no longer needs
`sroa,mem2reg,loop-simplify,lcssa` preprocessing (items 6.1 + 6.2
+ 6.3 above complete this), the final pipeline is `sroa,mem2reg`
only. At that point the RVSDG is the canonical form, and any
optimisation - constant folding, dead-code elimination,
loop-invariant code motion, the standard list - runs as a pass
over the RVSDG itself. The construction work described in this
report is upstream of all of that; the optimisation pipeline is a
separate body of work that does not touch `region/`.

## 7. Verification today

The construction is exercised by:

- **260 unit tests in `cargo test --lib`** covering RVSDG builder
  primitives, type interning, lowering of individual ops, and the
  `analyze_loop` classifications.
- **13 end-to-end JIT fixtures** under `examples/c/`, each
  compiled via the LLVM ingest pipeline through our construction,
  lowered back to LLVM, JIT-compiled, and compared against
  `clang -O0`'s exit code. The fixtures cover straight-line code,
  calls, if/else, do-while, while, loop-with-break, nested loops,
  switches, triple nesting, gamma-inside-loop, loop-inside-gamma,
  zero-iteration test-first, no-phi loops, and interior-join
  phi-use loops.

A failure in any of those flagged a real construction bug each
time during Phase 2 development; the current green state means
the constraints listed in section 5 are the only known gaps.
