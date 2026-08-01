//! **The scoped symbol table** -- the paper's symbol table (Bahmann et al.
//! 2015, section 4: "a symbol table to map each variable name in the CFG to
//! its definition place in the RVSDG") with one frame per RVSDG region on
//! the emission stack.
//!
//! The paper's construction is process-first, assemble-afterwards (its
//! section 4 branch rule): "process each alternative path; AFTERWARDS,
//! generate a gamma-node that uses ... all variables required in the
//! subregions as input ... update the symbol table with all variables
//! assigned to any of the alternate paths". This table makes both
//! observations as a side effect of the symbolic execution itself:
//!
//! - Resolving a symbol walks from its innermost binding outward. A binding
//!   found in an OUTER frame is captured through every intervening region:
//!   a parameter is appended to each region on the way in, each frame
//!   caches the capture, and the capture list of a frame IS the construct's
//!   "variables required" (its inputs).
//! - Writes land in the current frame with a written flag and a first-write
//!   order; a frame's writes ARE the construct's "variables assigned" (its
//!   outputs).
//!
//! There are deliberately NO static input/output scans anywhere: a scan is
//! a second source of truth about what the walk binds, and any disagreement
//! is a silent miscompile (this bit twice before this table existed).
//!
//! # Representation
//!
//! Symbols are interned to dense [`SymbolId`]s once per function, and every
//! lookup after interning is an array index: the table keeps one binding
//! STACK per symbol (entries tagged with the frame depth that bound them)
//! instead of one map per frame. Resolving is "look at the stack top";
//! binding pushes or updates the top; popping a frame pops exactly the
//! entries its `write_order` and `captures` lists name. This removed the
//! symbol-keyed hash probes that dominated construction profiles (LLVM
//! names hash string content; every operand of every instruction resolves).
//! Numbered LLVM names (`%0, %1, ...` -- nearly all locals under
//! `-disable-llvm-passes`) intern by direct array index with no hashing at
//! all.
//!
//! The memory state never lives here; it rides the builder's dedicated
//! state ports.

use llvm_ir::Name;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::{
    llvm_parser::control_flow::overlay::AuxVar,
    rvsdg::{RegionId, ValueId, function_graph::FunctionGraph},
};

/// A dense per-function handle for one symbol. All hot-path bookkeeping is
/// keyed by this id; the [`Symbol`] itself is stored once, at interning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::llvm_parser) struct SymbolId(u32);

/// The function's return value, interned first in every table.
pub(in crate::llvm_parser) const RET_VAL: SymbolId = SymbolId(0);

/// A symbol the emitter tracks across constructs: an LLVM SSA name, an
/// auxiliary selector invented by restructuring, or the function's return
/// value. Stored once per unique symbol; everything else uses [`SymbolId`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::llvm_parser) enum Symbol {
    Name(Name),
    Aux(AuxVar),
    RetVal,
}

/// One frame's view of a symbol: its current value in this frame's region,
/// and whether the frame WROTE it (an output) or merely captured it from an
/// outer frame (an input).
#[derive(Clone, Copy, Debug)]
pub(in crate::llvm_parser) struct FrameBinding {
    pub value: ValueId,
    pub written: bool,
}

/// One on-demand capture: `symbol` was read here but bound outside, so
/// `param` was appended to this frame's region and carries `outer` in.
#[derive(Clone, Copy, Debug)]
pub(in crate::llvm_parser) struct Capture {
    pub symbol: SymbolId,
    /// The symbol's value in the enclosing frame at capture time (the value
    /// the construct's input is wired to).
    pub outer: ValueId,
    pub param: ValueId,
}

/// One entry of a symbol's binding stack: the frame depth that bound it and
/// the binding itself. Entries are in ascending depth order, so the top is
/// always the innermost binding.
#[derive(Clone, Copy, Debug)]
struct StackEntry {
    depth: u32,
    binding: FrameBinding,
}

/// One region's symbol scope. While the frame is live its bindings live on
/// the per-symbol stacks; popping materialises them into `finals` so the
/// assembly phase can keep querying the frame after it left the stack.
#[derive(Debug)]
pub(in crate::llvm_parser) struct Frame {
    pub region: RegionId,
    /// Symbols this frame wrote, in first-write order: the construct's
    /// outputs, deterministically ordered.
    pub write_order: Vec<SymbolId>,
    /// Symbols this frame captured from outer frames: the construct's
    /// inputs.
    pub captures: Vec<Capture>,
    /// The frame's final bindings, filled at pop (empty while live).
    finals: FxHashMap<SymbolId, FrameBinding>,
}

impl Frame {
    fn empty(region: RegionId) -> Self {
        Self {
            region,
            write_order: Vec::new(),
            captures: Vec::new(),
            finals: FxHashMap::default(),
        }
    }

    /// The frame's final value for `symbol`, if it holds one. Only
    /// meaningful on a popped frame (assembly queries).
    pub(in crate::llvm_parser) fn final_value(&self, symbol: SymbolId) -> Option<FrameBinding> {
        self.finals.get(&symbol).copied()
    }
}

/// The scope stack. The root frame is the function region; gamma
/// alternatives and theta bodies push and pop frames around their regions.
#[derive(Debug)]
pub(in crate::llvm_parser) struct SymbolScopes {
    frames: Vec<Frame>,
    /// Recycled frame storage: cleared vectors and maps keep their
    /// capacity, so steady-state emission allocates nothing per frame.
    free_frames: Vec<Frame>,
    /// One binding stack per interned symbol, indexed by [`SymbolId`].
    stacks: Vec<SmallVec<[StackEntry; 2]>>,
    /// Reverse table: id -> symbol, one entry per unique symbol.
    symbols: Vec<Symbol>,
    /// Numbered LLVM names intern by direct index: `numbered[n]` is the
    /// id + 1 of `Name::Number(n)`, 0 while unassigned.
    numbered: Vec<u32>,
    /// String-named LLVM names (one hash per occurrence, on the name only).
    named: FxHashMap<Name, SymbolId>,
    /// Auxiliary restructuring selectors.
    aux: FxHashMap<AuxVar, SymbolId>,
}

impl SymbolScopes {
    pub(in crate::llvm_parser) fn new(root_region: RegionId) -> Self {
        Self {
            frames: vec![Frame::empty(root_region)],
            free_frames: Vec::new(),
            // Slot 0 is RET_VAL, matching the `RET_VAL` constant.
            stacks: vec![SmallVec::new()],
            symbols: vec![Symbol::RetVal],
            numbered: Vec::new(),
            named: FxHashMap::default(),
            aux: FxHashMap::default(),
        }
    }

    fn add_symbol(&mut self, symbol: Symbol) -> SymbolId {
        let id = SymbolId(self.symbols.len() as u32);
        self.symbols.push(symbol);
        self.stacks.push(SmallVec::new());
        id
    }

    /// Intern an LLVM name. Numbered names index an array directly; string
    /// names pay one hash.
    pub(in crate::llvm_parser) fn intern_name(&mut self, name: &Name) -> SymbolId {
        match name {
            Name::Number(n) => {
                if *n >= self.numbered.len() {
                    self.numbered.resize(n + 1, 0);
                }
                let slot = self.numbered[*n];
                if slot != 0 {
                    return SymbolId(slot - 1);
                }
                let id = self.add_symbol(Symbol::Name(name.clone()));
                self.numbered[*n] = id.0 + 1;
                id
            }
            Name::Name(_) => {
                if let Some(&id) = self.named.get(name) {
                    return id;
                }
                let id = self.add_symbol(Symbol::Name(name.clone()));
                self.named.insert(name.clone(), id);
                id
            }
        }
    }

    /// Intern an auxiliary restructuring selector.
    pub(in crate::llvm_parser) fn intern_aux(&mut self, var: AuxVar) -> SymbolId {
        if let Some(&id) = self.aux.get(&var) {
            return id;
        }
        let id = self.add_symbol(Symbol::Aux(var));
        self.aux.insert(var, id);
        id
    }

    pub(in crate::llvm_parser) fn push_frame(&mut self, region: RegionId) {
        match self.free_frames.pop() {
            Some(mut frame) => {
                frame.region = region;
                self.frames.push(frame);
            }
            None => self.frames.push(Frame::empty(region)),
        }
    }

    /// Pop the current frame, materialising its final bindings for the
    /// assembly phase and removing its entries from the binding stacks.
    pub(in crate::llvm_parser) fn pop_frame(&mut self) -> Frame {
        debug_assert!(self.frames.len() > 1, "popping the function root frame");
        let depth = (self.frames.len() - 1) as u32;
        let mut frame = self.frames.pop().expect("scope stack is never empty");
        // Every stack entry at this depth was recorded in exactly one of
        // the two lists (a write in write_order, a capture in captures); a
        // capture that was later written appears in both, so the second
        // visit finds its entry already popped and skips.
        for i in 0..frame.write_order.len() {
            let id = frame.write_order[i];
            Self::pop_entry(&mut self.stacks, &mut frame.finals, id, depth);
        }
        for i in 0..frame.captures.len() {
            let id = frame.captures[i].symbol;
            Self::pop_entry(&mut self.stacks, &mut frame.finals, id, depth);
        }
        frame
    }

    fn pop_entry(
        stacks: &mut [SmallVec<[StackEntry; 2]>],
        finals: &mut FxHashMap<SymbolId, FrameBinding>,
        id: SymbolId,
        depth: u32,
    ) {
        let stack = &mut stacks[id.0 as usize];
        if stack.last().is_some_and(|entry| entry.depth == depth) {
            let entry = stack.pop().expect("just checked");
            finals.insert(id, entry.binding);
        }
    }

    /// Return a popped frame's storage for reuse once assembly is done
    /// with it.
    pub(in crate::llvm_parser) fn recycle_frame(&mut self, mut frame: Frame) {
        frame.write_order.clear();
        frame.captures.clear();
        frame.finals.clear();
        self.free_frames.push(frame);
    }

    /// Bind `id` in the current frame (a write: the symbol becomes an
    /// output of the enclosing construct).
    pub(in crate::llvm_parser) fn bind_id(&mut self, id: SymbolId, value: ValueId) {
        let depth = (self.frames.len() - 1) as u32;
        let frame = self.frames.last_mut().expect("scope stack is never empty");
        let stack = &mut self.stacks[id.0 as usize];
        let binding = FrameBinding {
            value,
            written: true,
        };
        match stack.last_mut() {
            Some(entry) if entry.depth == depth => {
                if !entry.binding.written {
                    frame.write_order.push(id);
                }
                entry.binding = binding;
            }
            _ => {
                frame.write_order.push(id);
                stack.push(StackEntry { depth, binding });
            }
        }
    }

    /// Convenience for the hot path: bind an LLVM name.
    pub(in crate::llvm_parser) fn bind_name(&mut self, name: &Name, value: ValueId) {
        let id = self.intern_name(name);
        self.bind_id(id, value);
    }

    /// Bind an auxiliary selector.
    pub(in crate::llvm_parser) fn bind_aux(&mut self, var: AuxVar, value: ValueId) {
        let id = self.intern_aux(var);
        self.bind_id(id, value);
    }

    /// Resolve `id` for a read in the current region. A binding below the
    /// current depth is captured through every region in between: each gets
    /// a new parameter carrying the value inward, and each frame records
    /// the capture so later reads reuse it.
    pub(in crate::llvm_parser) fn resolve_id(
        &mut self,
        graph: &mut FunctionGraph,
        id: SymbolId,
    ) -> Option<ValueId> {
        let current = (self.frames.len() - 1) as u32;
        let top = *self.stacks[id.0 as usize].last()?;
        if top.depth == current {
            return Some(top.binding.value);
        }
        let mut value = top.binding.value;
        for depth in top.depth + 1..=current {
            let region = self.frames[depth as usize].region;
            let ty = graph.get_value_type(value);
            let param = graph.append_region_param(region, *ty);
            self.frames[depth as usize].captures.push(Capture {
                symbol: id,
                outer: value,
                param,
            });
            self.stacks[id.0 as usize].push(StackEntry {
                depth,
                binding: FrameBinding {
                    value: param,
                    written: false,
                },
            });
            value = param;
        }
        Some(value)
    }

    /// Resolve an LLVM name (the instruction-operand hot path).
    pub(in crate::llvm_parser) fn resolve_name(
        &mut self,
        graph: &mut FunctionGraph,
        name: &Name,
    ) -> Option<ValueId> {
        let id = self.intern_name(name);
        self.resolve_id(graph, id)
    }

    /// Resolve an auxiliary selector.
    pub(in crate::llvm_parser) fn resolve_aux(
        &mut self,
        graph: &mut FunctionGraph,
        var: AuxVar,
    ) -> Option<ValueId> {
        let id = self.intern_aux(var);
        self.resolve_id(graph, id)
    }
}
