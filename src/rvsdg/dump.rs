//! Human-readable debug dump for the RVSDG.
//!
//! This is a lossy view for debugging, not a serialization format -- it does
//! not carry enough to reconstruct the module. The design goals, in priority
//! order:
//!
//!   1. Every operand shows where it comes from. RVSDG is a dataflow graph, so
//!      the dump resolves each operand to its origin: a region argument prints
//!      as `aN` (the N-th parameter of the enclosing region), a structural
//!      node output prints as `%vNODE#K` (the K-th output of node NODE), and
//!      anything else prints as its value id `%vN`.
//!   2. The argument/result correspondence of gamma and theta nodes is
//!      explicit. A structural node lists its `in [ .. ]` operands (resolved in
//!      the enclosing region), each region declares its `args [ aN: ty ]`
//!      positionally, and the node's outputs are declared on the closing line
//!      as `} -> ( %vNODE#K: ty )`.
//!   3. `project` nodes are folded away. Their value is referenced inline as
//!      `%vNODE#K` at every use site, so the dump never prints a project line.
//!
//! Operand-origin rule: a use of `aN` always means "argument N of the region
//! the use appears in". Operands in a node's `in [ .. ]` list are resolved one
//! level out (the enclosing region), because that is the scope that feeds the
//! node. This holds because values never cross region boundaries directly;
//! they flow in through node inputs that become region arguments.

use std::fmt::{Display, Write};

use rustc_hash::FxHashSet;

use crate::rvsdg::{
    BinaryOp, ConstId, ConstValue, ConstantKind, FuncId, GlobalId, ICmpPred, RVSDGMod, RegionId,
    State, ValueId, ValueKind,
    func::Function,
    types::{
        ArrayTypeId, FuncTypeId, PtrType, ScalarType, StructId, TypeArena, TypeRef, VectorTypeId,
    },
};

/// How a region's body terminates: a top-level function region `return`s, a
/// gamma arm or theta body `yield`s its results back to the structural node.
enum Terminator {
    Return,
    Yield,
}

impl Terminator {
    fn keyword(&self) -> &'static str {
        match self {
            Terminator::Return => "return",
            Terminator::Yield => "yield",
        }
    }
}

impl Display for RVSDGMod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_constant_pool(f)?;
        for func in &self.functions {
            func.fmt(f, self)?;
        }
        Ok(())
    }
}

impl RVSDGMod {
    fn fmt_constant_pool(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("constants {\n")?;
        // Aggregate element constants are listed inline inside their parent
        // aggregate, so skip them as standalone entries.
        let mut agg_elements = FxHashSet::default();
        for id in self.constants.id_pool.0.iter() {
            agg_elements.insert(id.0);
        }
        for (i, const_def) in self.constants.entries.iter().enumerate() {
            if agg_elements.contains(&(i as u32)) {
                continue;
            }
            pad(f, 2)?;
            write!(f, "%c{i}: ")?;
            const_def.ty.fmt(f, &self.types)?;
            f.write_str(" = ")?;
            self.fmt_const_kind(f, &const_def.kind)?;
            f.write_char('\n')?;
        }
        f.write_str("}\n\n")?;
        Ok(())
    }

    fn fmt_const_kind(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        kind: &ConstantKind,
    ) -> std::fmt::Result {
        match kind {
            ConstantKind::Scalar(const_value) => write!(f, "{const_value}"),
            ConstantKind::Zero => f.write_str("zero"),
            ConstantKind::Aggregate(span) => {
                f.write_str("aggregate (")?;
                let elements = self.constants.get_aggregate_elements(*span);
                for (i, const_id) in elements.iter().enumerate() {
                    if i != 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{const_id}")?;
                }
                f.write_char(')')
            }
            ConstantKind::String(_) => f.write_str("string"),
            ConstantKind::GlobalAddr(global_id) => write!(f, "global_addr {global_id}"),
            ConstantKind::FuncAddr(func_id) => write!(f, "func_addr {func_id}"),
            ConstantKind::Undef => f.write_str("undef"),
            ConstantKind::GetElementPointer {
                base,
                source_type,
                in_bounds,
                ..
            } => {
                write!(f, "get_element_ptr base: {base}, ty: ")?;
                source_type.fmt(f, &self.types)?;
                write!(f, ", in_bounds: {in_bounds}")
            }
            ConstantKind::Cast { op, operand } => write!(f, "cast {op:?} {operand}"),
        }
    }
}

impl Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, m: &RVSDGMod) -> std::fmt::Result {
        write!(f, "fn {} {} (", self.id, self.name)?;
        for (i, param) in self.params.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }
            write!(f, "a{i}: ")?;
            param.ty.fmt(f, &m.types)?;
        }
        f.write_str(") -> (")?;
        for (i, ret_ty) in self.return_types.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }
            ret_ty.fmt(f, &m.types)?;
        }
        f.write_str(") {\n")?;

        match self.lambda_val {
            Some(lambda_id) => {
                let region_id = match m.get_value_kind(lambda_id) {
                    ValueKind::Lambda { region, .. } => region,
                    // A function's lambda_val must be a Lambda; anything else
                    // is a construction bug, so surface it rather than hide it.
                    ref other => {
                        write!(
                            f,
                            "    <malformed lambda: {other:?}>\n}} fn end {}\n\n",
                            self.id
                        )?;
                        return Ok(());
                    }
                };
                fmt_region_body(f, m, *region_id, 4, false, true, None, Terminator::Return)?;
            }
            None => pad(f, 4).and_then(|_| f.write_str("; external (no body)\n"))?,
        }

        write!(f, "}} fn end {}\n\n", self.id)?;
        Ok(())
    }
}

/// Print the body of a region: an optional incoming-state line, an optional
/// `args` line, the region's nodes, an optional `continue_if` predicate (theta
/// only), and the terminator line.
///
/// `indent` is the column at which each body line starts. `print_args`
/// controls the `args [ .. ]` line (suppressed for function-body regions,
/// whose parameters already appear in the signature). `show_entry_state`
/// prints the region's incoming state edge; it is set only for function-body
/// regions, since a nested region's incoming state is already shown on the
/// owning gamma/theta node as `state_in`. `continue_if` is the theta
/// repetition predicate, printed just before the terminator.
fn fmt_region_body(
    f: &mut std::fmt::Formatter<'_>,
    m: &RVSDGMod,
    region_id: RegionId,
    indent: usize,
    print_args: bool,
    show_entry_state: bool,
    continue_if: Option<ValueId>,
    terminator: Terminator,
) -> std::fmt::Result {
    let region = m.get_region(region_id);

    if show_entry_state {
        pad(f, indent)?;
        write!(f, "state_in {}\n", region.entry_state)?;
    }

    if print_args && !region.params.is_empty() {
        pad(f, indent)?;
        f.write_str("args [ ")?;
        for (i, &param_id) in region.params.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }
            let param = m.get_value_type(param_id);
            write!(f, "a{i}: ")?;
            param.fmt(f, &m.types)?;
        }
        f.write_str(" ]\n")?;
    }

    for &node_id in &region.nodes {
        // Structural scaffolding never prints as a body line: the lambda owns
        // this region, region params are shown in `args`, projects are folded
        // into `%vNODE#K` references, and the region result is the terminator.
        if matches!(
            m.get_value_kind(node_id),
            ValueKind::Lambda { .. }
                | ValueKind::Project { .. }
                | ValueKind::RegionParam { .. }
                | ValueKind::RegionResult { .. }
        ) {
            continue;
        }
        pad(f, indent)?;
        write!(f, "{node_id} = ")?;
        fmt_node(f, m, node_id, indent)?;
        f.write_char('\n')?;
    }

    if let Some(pred) = continue_if {
        pad(f, indent)?;
        f.write_str("continue_if ")?;
        fmt_value_ref(f, m, pred)?;
        f.write_char('\n')?;
    }

    pad(f, indent)?;
    write!(f, "{} [ ", terminator.keyword())?;
    for (i, &result) in m.value_pool.get(region.results).iter().enumerate() {
        if i != 0 {
            f.write_str(", ")?;
        }
        fmt_value_ref(f, m, result)?;
    }
    f.write_str(" ]\n")?;

    Ok(())
}

/// Print the right-hand side of a node definition (everything after `%vN = `).
/// `indent` is the column the `%vN =` started at, used to lay out the nested
/// blocks of structural nodes.
fn fmt_node(
    f: &mut std::fmt::Formatter<'_>,
    m: &RVSDGMod,
    node_id: ValueId,
    indent: usize,
) -> std::fmt::Result {
    let value = m.get_value_kind(node_id);
    match value {
        ValueKind::Const(const_value) => write!(f, "const {const_value}"),
        ValueKind::ConstPoolRef(const_id) => fmt_const_ref(f, m, *const_id),
        ValueKind::GlobalRef(global_id) => write!(f, "global_ref {global_id}"),
        ValueKind::FuncAddr(func_id) => write!(f, "func_addr {func_id}"),
        ValueKind::Binary {
            op, left, right, ..
        } => {
            write!(f, "{op} ")?;
            fmt_value_ref(f, m, *left)?;
            f.write_str(", ")?;
            fmt_value_ref(f, m, *right)
        }
        ValueKind::ICmp { pred, left, right } => {
            write!(f, "icmp {} ", icmp_pred_str(*pred))?;
            fmt_value_ref(f, m, *left)?;
            f.write_str(", ")?;
            fmt_value_ref(f, m, *right)
        }
        ValueKind::Match {
            input,
            arms,
            default,
            ..
        } => {
            f.write_str("match ")?;
            fmt_value_ref(f, m, *input)?;
            f.write_str(" { ")?;
            for arm in m.match_arm_pool.get(*arms) {
                write!(f, "{} => arm{}, ", arm.value, arm.alternative)?;
            }
            write!(f, "_ => arm{default} }}")
        }
        ValueKind::Call {
            state,
            fn_id,
            sig,
            args,
        } => {
            write!(f, "call {fn_id} {sig} args ")?;
            fmt_value_list(f, m, *args)?;
            write!(f, " state_in {state} state_out %s{}", node_id.0)
        }
        ValueKind::CallIndirect {
            state,
            callee,
            sig,
            args,
        } => {
            write!(f, "call_indirect {sig} callee ")?;
            fmt_value_ref(f, m, *callee)?;
            f.write_str(" args ")?;
            fmt_value_list(f, m, *args)?;
            write!(f, " state_in {state} state_out %s{}", node_id.0)
        }
        ValueKind::Gamma {
            condition,
            inputs,
            state,
            regions,
        } => {
            f.write_str("gamma predicate ")?;
            fmt_value_ref(f, m, *condition)?;
            write!(f, " state_in {state} state_out %s{} {{\n", node_id.0)?;

            pad(f, indent + 4)?;
            f.write_str("in ")?;
            fmt_value_list(f, m, *inputs)?;
            f.write_char('\n')?;

            let region_ids = m.region_pool.get(*regions);
            for (i, &arm_region) in region_ids.iter().enumerate() {
                pad(f, indent + 4)?;
                write!(f, "arm{i}:\n")?;
                fmt_region_body(
                    f,
                    m,
                    arm_region,
                    indent + 8,
                    true,
                    false,
                    None,
                    Terminator::Yield,
                )?;
            }

            // All arms share output arity/types; take them from the first arm.
            let outputs = m.get_region(region_ids[0]).results;
            fmt_struct_outputs(f, m, node_id, outputs, indent)
        }
        ValueKind::Theta {
            loop_vars,
            condition,
            state,
            region_id,
        } => {
            write!(f, "theta state_in {state} state_out %s{} {{\n", node_id.0)?;

            pad(f, indent + 4)?;
            f.write_str("in ")?;
            fmt_value_list(f, m, *loop_vars)?;
            f.write_char('\n')?;

            fmt_region_body(
                f,
                m,
                *region_id,
                indent + 4,
                true,
                false,
                Some(*condition),
                Terminator::Yield,
            )?;

            let outputs = m.get_region(*region_id).results;
            fmt_struct_outputs(f, m, node_id, outputs, indent)
        }
        // Remaining kinds are not yet given a bespoke rendering. Fall back to a
        // debug print so the dump degrades gracefully instead of panicking on
        // inputs that exercise them.
        other => write!(f, "{other:?}"),
    }
}

/// Print a structural node's outputs on its closing line:
/// `} -> ( %vNODE#0: ty, %vNODE#1: ty )`. `result_values` are the region
/// results whose types and count define the outputs.
fn fmt_struct_outputs(
    f: &mut std::fmt::Formatter<'_>,
    m: &RVSDGMod,
    node_id: ValueId,
    result_values: crate::rvsdg::ValuesSpan,
    indent: usize,
) -> std::fmt::Result {
    pad(f, indent)?;
    f.write_str("} -> ( ")?;
    for (i, &result) in m.value_pool.get(result_values).iter().enumerate() {
        if i != 0 {
            f.write_str(", ")?;
        }
        write!(f, "%v{}#{i}: ", node_id.0)?;
        m.get_value_type(result).fmt(f, &m.types)?;
    }
    f.write_str(" )")
}

/// Print a `[ op, op, .. ]` list of operands, each resolved to its origin.
fn fmt_value_list(
    f: &mut std::fmt::Formatter<'_>,
    m: &RVSDGMod,
    span: crate::rvsdg::ValuesSpan,
) -> std::fmt::Result {
    f.write_str("[ ")?;
    for (i, &id) in m.value_pool.get(span).iter().enumerate() {
        if i != 0 {
            f.write_str(", ")?;
        }
        fmt_value_ref(f, m, id)?;
    }
    f.write_str(" ]")
}

/// Resolve a value reference to its origin and print it: a region argument as
/// `aN`, a folded project output as `%vNODE#K`, everything else as `%vN`.
fn fmt_value_ref(f: &mut std::fmt::Formatter<'_>, m: &RVSDGMod, id: ValueId) -> std::fmt::Result {
    match m.get_value_kind(id) {
        ValueKind::RegionParam { index, .. } => write!(f, "a{index}"),
        ValueKind::Project { call, index } => write!(f, "%v{}#{}", call.0, index),
        // Region-free values have no defining line in any region (they
        // are interned module-wide), so print them inline at each use.
        ValueKind::Const(const_value) => write!(f, "const {const_value}"),
        ValueKind::GlobalRef(global_id) => write!(f, "global_addr {global_id}"),
        ValueKind::FuncAddr(func_id) => write!(f, "func_addr {func_id}"),
        ValueKind::ConstPoolRef(const_id) => fmt_const_ref(f, m, *const_id),
        _ => write!(f, "{id}"),
    }
}

/// Print the right-hand side of a `ConstPoolRef` node, resolved through the
/// constant pool so the reader sees the value rather than a `%cN` indirection.
fn fmt_const_ref(
    f: &mut std::fmt::Formatter<'_>,
    m: &RVSDGMod,
    const_id: ConstId,
) -> std::fmt::Result {
    match &m.constants.get(const_id).kind {
        ConstantKind::Scalar(const_value) => write!(f, "const {const_value}"),
        ConstantKind::FuncAddr(func_id) => write!(f, "func_addr {func_id}"),
        ConstantKind::GlobalAddr(global_id) => write!(f, "global_addr {global_id}"),
        ConstantKind::Zero => f.write_str("const zero"),
        // Aggregates, strings, pointer offsets etc. keep the pool reference;
        // the value itself is spelled out in the constants section.
        _ => write!(f, "const_pool_ref {const_id}"),
    }
}

fn icmp_pred_str(pred: ICmpPred) -> &'static str {
    match pred {
        ICmpPred::Eq => "eq",
        ICmpPred::Ne => "ne",
        ICmpPred::UnsignedGt => "unsigned_gt",
        ICmpPred::UnsignedGe => "unsigned_ge",
        ICmpPred::UnsignedLt => "unsigned_lt",
        ICmpPred::UnsignedLe => "unsigned_le",
        ICmpPred::SignedGt => "signed_gt",
        ICmpPred::SignedGe => "signed_ge",
        ICmpPred::SignedLt => "signed_lt",
        ICmpPred::SignedLe => "signed_le",
    }
}

fn pad(f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
    for _ in 0..indent {
        f.write_char(' ')?;
    }
    Ok(())
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ScalarType::IntArbitrary(bits) => return write!(f, "i{bits}"),
            ScalarType::Bool => "bool",
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::I128 => "i128",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::F80 => "f80",
            ScalarType::Void => "void",
        };
        f.write_str(s)
    }
}

impl TypeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, type_arena: &TypeArena) -> std::fmt::Result {
        match self {
            TypeRef::State => f.write_str("state"),
            TypeRef::Scalar(scalar_type) => write!(f, "{scalar_type}"),
            TypeRef::Ptr(ptr_type_id) => type_arena.get_ptr(*ptr_type_id).fmt(f, type_arena),
            TypeRef::Array(array_type_id) => write!(f, "{array_type_id}"),
            TypeRef::Struct(struct_id) => write!(f, "{struct_id}"),
            TypeRef::Vector(vector_type_id) => write!(f, "{vector_type_id}"),
            TypeRef::Func(func_type_id) => write!(f, "{func_type_id}"),
            TypeRef::Control(n) => write!(f, "control{n}"),
        }
    }
}

impl PtrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, type_arena: &TypeArena) -> std::fmt::Result {
        f.write_str("ptr ")?;
        match self.pointee {
            Some(ty) => ty.fmt(f, type_arena),
            None => f.write_str("opaque"),
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::SignedDiv => "signed_div",
            BinaryOp::UnsignedDiv => "unsigned_div",
            BinaryOp::SignedRem => "signed_rem",
            BinaryOp::UnsignedRem => "unsigned_rem",
            BinaryOp::ShiftLeft => "shift_left",
            BinaryOp::LogicalShiftRight => "logical_shift_right",
            BinaryOp::ArithShiftRight => "arith_shift_right",
            BinaryOp::And => "and",
            BinaryOp::Or => "or",
            BinaryOp::Xor => "xor",
            BinaryOp::FloatAdd => "float_add",
            BinaryOp::FloatSub => "float_sub",
            BinaryOp::FloatMul => "float_mul",
            BinaryOp::FloatDiv => "float_div",
            BinaryOp::FloatRem => "float_rem",
        };
        f.write_str(s)
    }
}

impl Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%r{}", self.0)
    }
}
impl Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%s{}", self.0.0)
    }
}

impl Display for FuncId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%f{}", self.0)
    }
}

impl Display for GlobalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%g{}", self.0)
    }
}

impl Display for ConstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%c{}", self.0)
    }
}

impl Display for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::Int(val) => write!(f, "{val}"),
            ConstValue::F32(bits) => write!(f, "{}", f32::from_bits(*bits)),
            ConstValue::F64(bits) => write!(f, "{}", f64::from_bits(*bits)),
            ConstValue::NullPtr => f.write_str("null"),
            ConstValue::Poison => f.write_str("poison"),
        }
    }
}

impl Display for ArrayTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%arr{}", self.0)
    }
}

impl Display for StructId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%struct{}", self.0)
    }
}

impl Display for VectorTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%vec{}", self.0)
    }
}

impl Display for FuncTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%func{}", self.0)
    }
}
