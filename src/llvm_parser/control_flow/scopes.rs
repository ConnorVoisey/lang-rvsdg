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
//! - Resolving a symbol walks frames innermost to outermost. A binding
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
//! The memory state never lives here; it rides the builder's dedicated
//! state ports.

use llvm_ir::Name;
use rustc_hash::FxHashMap;

use crate::{
    llvm_parser::control_flow::overlay::AuxVar,
    rvsdg::{RVSDGMod, RegionId, ValueId},
};

/// A symbol the emitter tracks across constructs: an LLVM SSA name, an
/// auxiliary selector invented by restructuring, or the function's return
/// value.
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
#[derive(Clone, Debug)]
pub(in crate::llvm_parser) struct Capture {
    pub symbol: Symbol,
    /// The symbol's value in the enclosing frame at capture time (the value
    /// the construct's input is wired to).
    pub outer: ValueId,
    pub param: ValueId,
}

/// One region's symbol scope. Bindings are split by symbol kind so LLVM
/// name lookups (the hot path: every instruction operand) need no `Symbol`
/// allocation.
#[derive(Debug)]
pub(in crate::llvm_parser) struct Frame {
    pub region: RegionId,
    names: FxHashMap<Name, FrameBinding>,
    aux: FxHashMap<AuxVar, FrameBinding>,
    ret_val: Option<FrameBinding>,
    /// Symbols this frame wrote, in first-write order: the construct's
    /// outputs, deterministically ordered.
    pub write_order: Vec<Symbol>,
    /// Symbols this frame captured from outer frames: the construct's
    /// inputs.
    pub captures: Vec<Capture>,
}

impl Frame {
    fn empty(region: RegionId) -> Self {
        Self {
            region,
            names: FxHashMap::default(),
            aux: FxHashMap::default(),
            ret_val: None,
            write_order: Vec::new(),
            captures: Vec::new(),
        }
    }

    fn binding(&self, symbol: &Symbol) -> Option<FrameBinding> {
        match symbol {
            Symbol::Name(name) => self.names.get(name).copied(),
            Symbol::Aux(var) => self.aux.get(var).copied(),
            Symbol::RetVal => self.ret_val,
        }
    }

    fn set_binding(&mut self, symbol: &Symbol, binding: FrameBinding) {
        match symbol {
            Symbol::Name(name) => {
                self.names.insert(name.clone(), binding);
            }
            Symbol::Aux(var) => {
                self.aux.insert(*var, binding);
            }
            Symbol::RetVal => self.ret_val = Some(binding),
        }
    }

    /// The frame's final value for `symbol`, if it holds one.
    pub(in crate::llvm_parser) fn final_value(&self, symbol: &Symbol) -> Option<FrameBinding> {
        self.binding(symbol)
    }
}

/// The scope stack. The root frame is the function region; gamma
/// alternatives and theta bodies push and pop frames around their regions.
#[derive(Debug)]
pub(in crate::llvm_parser) struct SymbolScopes {
    frames: Vec<Frame>,
}

impl SymbolScopes {
    pub(in crate::llvm_parser) fn new(root_region: RegionId) -> Self {
        Self {
            frames: vec![Frame::empty(root_region)],
        }
    }

    pub(in crate::llvm_parser) fn push_frame(&mut self, region: RegionId) {
        self.frames.push(Frame::empty(region));
    }

    pub(in crate::llvm_parser) fn pop_frame(&mut self) -> Frame {
        debug_assert!(self.frames.len() > 1, "popping the function root frame");
        self.frames.pop().expect("scope stack is never empty")
    }

    /// Bind `symbol` in the current frame (a write: the symbol becomes an
    /// output of the enclosing construct).
    pub(in crate::llvm_parser) fn bind(&mut self, symbol: Symbol, value: ValueId) {
        let frame = self.frames.last_mut().expect("scope stack is never empty");
        let already_written = frame
            .binding(&symbol)
            .is_some_and(|binding| binding.written);
        if !already_written {
            frame.write_order.push(symbol.clone());
        }
        frame.set_binding(
            &symbol,
            FrameBinding {
                value,
                written: true,
            },
        );
    }

    /// Convenience for the hot path: bind an LLVM name.
    pub(in crate::llvm_parser) fn bind_name(&mut self, name: Name, value: ValueId) {
        self.bind(Symbol::Name(name), value);
    }

    /// Resolve `symbol` for a read in the current region. A binding in an
    /// outer frame is captured through every region in between: each gets a
    /// new parameter carrying the value inward, and each frame caches the
    /// capture so later reads reuse it.
    pub(in crate::llvm_parser) fn resolve(
        &mut self,
        graph: &mut RVSDGMod,
        symbol: &Symbol,
    ) -> Option<ValueId> {
        let found_depth = self
            .frames
            .iter()
            .rposition(|frame| frame.binding(symbol).is_some())?;
        let mut value = self.frames[found_depth]
            .binding(symbol)
            .expect("just located")
            .value;
        for depth in found_depth + 1..self.frames.len() {
            let region = self.frames[depth].region;
            let ty = graph.values[value.0 as usize].ty;
            let param = graph.append_region_param(region, ty);
            self.frames[depth].captures.push(Capture {
                symbol: symbol.clone(),
                outer: value,
                param,
            });
            self.frames[depth].set_binding(
                symbol,
                FrameBinding {
                    value: param,
                    written: false,
                },
            );
            value = param;
        }
        Some(value)
    }

    /// Resolve an LLVM name (the instruction-operand hot path).
    pub(in crate::llvm_parser) fn resolve_name(
        &mut self,
        graph: &mut RVSDGMod,
        name: &Name,
    ) -> Option<ValueId> {
        // Fast path: bound in the current frame already.
        if let Some(binding) = self
            .frames
            .last()
            .expect("scope stack is never empty")
            .names
            .get(name)
        {
            return Some(binding.value);
        }
        self.resolve(graph, &Symbol::Name(name.clone()))
    }
}
