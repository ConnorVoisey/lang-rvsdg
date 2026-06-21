use std::fmt::{Debug, Display, Write};

use rustc_hash::FxHashSet;

use crate::rvsdg::{
    BinaryOp, ConstValue, ConstantKind, ConstantPool, RVSDGMod, Region, RegionId, Value, ValueId,
    ValueKind,
    func::Function,
    types::{PtrType, ScalarType, TypeArena, TypeRef},
};

impl Display for RVSDGMod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_constant_pool(f)?;
        for func in &self.functions {
            func.fmt(f, &self)?;
        }
        Ok(())
    }
}

impl RVSDGMod {
    fn fmt_constant_pool(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("constants {\n")?;
        let mut agg_set = FxHashSet::default();
        for id in self.constants.id_pool.0.iter() {
            agg_set.insert(id.0);
        }
        for (i, const_def) in self.constants.entries.iter().enumerate() {
            if agg_set.contains(&(i as u32)) {
                continue;
            }
            apply_indent(f, 2)?;
            write!(f, "%c{} ", i)?;
            match &const_def.kind {
                ConstantKind::Scalar(const_value) => match const_value {
                    ConstValue::Int(val) => match const_def.ty {
                        TypeRef::Scalar(ScalarType::I8) => write!(f, "i8 {val}")?,
                        TypeRef::Scalar(ScalarType::I16) => write!(f, "i16 {val}")?,
                        TypeRef::Scalar(ScalarType::I32) => write!(f, "i32 {val}")?,
                        TypeRef::Scalar(ScalarType::I64) => write!(f, "i64 {val}")?,
                        TypeRef::Scalar(ScalarType::I128) => write!(f, "i128 {val}")?,
                        t => unreachable!("int const doesn't have int type: {t:?}"),
                    },
                    ConstValue::F32(val) => write!(f, "f32 {}", f32::from_bits(*val as u32))?,
                    ConstValue::F64(val) => write!(f, "f64 {}", f64::from_bits(*val as u64))?,
                    ConstValue::NullPtr => todo!(),
                    ConstValue::Poison => todo!(),
                },
                ConstantKind::Zero => todo!(),
                ConstantKind::Aggregate(const_ids_span) => {
                    write!(f, "aggregate (")?;
                    let aggs = self.constants.get_aggregate_elements(*const_ids_span);
                    for (i, const_id) in aggs.iter().enumerate() {
                        write!(f, "%c{}", const_id.0)?;
                        if i != aggs.len() - 1 {
                            f.write_str(", ")?;
                        }
                    }
                    f.write_char(')')?;
                }
                ConstantKind::String(items) => todo!(),
                ConstantKind::GlobalAddr(global_id) => todo!(),
                ConstantKind::FuncAddr(func_id) => todo!(),
                ConstantKind::Undef => todo!(),
                ConstantKind::GetElementPointer {
                    base,
                    source_type,
                    indices,
                    in_bounds,
                } => todo!(),
            }
            f.write_char('\n')?;
        }
        f.write_str("} end constants\n\n")?;
        Ok(())
    }
}

impl Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, rvsdg_mod: &RVSDGMod) -> std::fmt::Result {
        write!(f, "fn %{} name: {} (", self.id.0, self.name)?;
        for (i, param) in self.params.iter().enumerate() {
            param.ty.fmt(f, &rvsdg_mod.types)?;
            if i != self.params.len() - 1 {
                f.write_str(", ")?;
            }
        }
        f.write_str(") -> (")?;
        for (i, ret_ty) in self.return_types.iter().enumerate() {
            ret_ty.fmt(f, &rvsdg_mod.types)?;
            if i != self.return_types.len() - 1 {
                f.write_str(", ")?;
            }
        }
        f.write_str(") {\n")?;
        match self.lambda_val {
            Some(lam_val) => {
                rvsdg_mod.get(lam_val).fmt(f, rvsdg_mod)?;
            }
            None => f.write_str("EMPTY FN BODY\n")?,
        }

        write!(f, "}} fn end %{}\n\n", self.id.0)?;
        Ok(())
    }
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarType::Bool => f.write_str("Bool"),
            ScalarType::I8 => f.write_str("I8"),
            ScalarType::I16 => f.write_str("I16"),
            ScalarType::I32 => f.write_str("I32"),
            ScalarType::I64 => f.write_str("I64"),
            ScalarType::I128 => f.write_str("I128"),
            ScalarType::F32 => f.write_str("F32"),
            ScalarType::F64 => f.write_str("F64"),
            ScalarType::Void => f.write_str("Void"),
        }
    }
}

impl TypeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, type_arena: &TypeArena) -> std::fmt::Result {
        match self {
            TypeRef::State => f.write_str("State"),
            TypeRef::Scalar(scalar_type) => write!(f, "{}", scalar_type),
            TypeRef::Ptr(ptr_type_id) => type_arena.get_ptr(*ptr_type_id).fmt(f, type_arena),
            TypeRef::Array(array_type_id) => todo!(),
            TypeRef::Struct(struct_id) => todo!(),
            TypeRef::Vector(vector_type_id) => todo!(),
            TypeRef::Func(func_type_id) => todo!(),
            TypeRef::Control(_) => todo!(),
        }
    }
}
impl PtrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, type_arena: &TypeArena) -> std::fmt::Result {
        match self.pointee {
            Some(ty) => ty.fmt(f, type_arena),
            None => f.write_str("None"),
        }
    }
}
impl Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, rvsdg_mod: &RVSDGMod) -> std::fmt::Result {
        match self.kind {
            ValueKind::Const(const_value) => todo!(),
            ValueKind::ConstPoolRef(const_id) => write!(f, "const_pool_ref %{}", const_id.0)?,
            ValueKind::GlobalRef(global_id) => todo!(),
            ValueKind::FuncAddr(func_id) => todo!(),
            ValueKind::Unary { op, operand } => todo!(),
            ValueKind::Binary {
                op, left, right, ..
            } => {
                write!(f, "{op} %{} %{}", left.0, right.0)?;
            }
            ValueKind::ICmp { pred, left, right } => todo!(),
            ValueKind::FCmp { pred, left, right } => todo!(),
            ValueKind::Ternary {
                condition,
                true_val,
                false_val,
            } => todo!(),
            ValueKind::Cast { op, value } => todo!(),
            ValueKind::ExtractLane { vector, index } => todo!(),
            ValueKind::InsertLane {
                vector,
                index,
                value,
            } => todo!(),
            ValueKind::ShuffleLanes { left, right, mask } => todo!(),
            ValueKind::ExtractField { aggregate, indices } => todo!(),
            ValueKind::InsertField {
                aggregate,
                value,
                indices,
            } => todo!(),
            ValueKind::PtrOffset {
                base,
                base_type,
                indices,
                inbounds,
            } => todo!(),
            ValueKind::Load {
                state,
                addr,
                loaded_type,
                align,
                volatile,
            } => todo!(),
            ValueKind::Store {
                state,
                addr,
                value,
                align,
                volatile,
            } => todo!(),
            ValueKind::Alloca {
                state,
                elem_type,
                count,
            } => todo!(),
            ValueKind::AtomicLoad {
                state,
                addr,
                loaded_type,
                ordering,
                align,
            } => todo!(),
            ValueKind::AtomicStore {
                state,
                addr,
                value,
                ordering,
                align,
            } => todo!(),
            ValueKind::AtomicReadModifyWrite {
                state,
                addr,
                value,
                op,
                ordering,
            } => todo!(),
            ValueKind::CompareAndSwap {
                state,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            } => todo!(),
            ValueKind::Fence { state, ordering } => todo!(),
            ValueKind::Freeze { value } => todo!(),
            ValueKind::Match {
                input,
                arms,
                default,
                alternatives,
            } => todo!(),
            ValueKind::Intrinsic { op, state, args } => todo!(),
            ValueKind::Lambda { region, func_id } => {
                let region = rvsdg_mod.get_region(region);
                region.fmt(f, 2, rvsdg_mod)?;
                // write!(f, "lambda func: %{} region: %{}", region.0, func_id.0)?
            }
            ValueKind::Theta {
                loop_vars,
                condition,
                state,
                region_id,
            } => todo!(),
            ValueKind::Gamma {
                condition,
                inputs,
                state,
                regions,
            } => todo!(),
            ValueKind::Phi { region, rv_count } => todo!(),
            ValueKind::Call { state, fn_id, args } => todo!(),
            ValueKind::CallIndirect {
                state,
                callee,
                args,
            } => write!(
                f,
                "call_indirect state: %{} callee: %{} args_start: {} args_len: {}",
                state.0.0, callee.0, args.start, args.len
            )?,
            ValueKind::Project { call, index } => {
                write!(f, "project call: %{} index: {}", call.0, index)?
            }
            ValueKind::RegionParam { index, ty } => {
                write!(f, "region_param {index} ")?;
                ty.fmt(f, &rvsdg_mod.types)?;
            }
            ValueKind::RegionResult { values, state } => todo!(),
        }
        Ok(())
    }
}
impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => f.write_str("add"),
            BinaryOp::Sub => f.write_str("Sub"),
            BinaryOp::Mul => f.write_str("mul"),
            BinaryOp::SignedDiv => f.write_str("signed_div"),
            BinaryOp::UnsignedDiv => f.write_str("unsigned_div"),
            BinaryOp::SignedRem => f.write_str("signed_rem"),
            BinaryOp::UnsignedRem => f.write_str("unsigned_rem"),
            BinaryOp::ShiftLeft => f.write_str("shift_left"),
            BinaryOp::LogicalShiftRight => f.write_str("logical_shift_right"),
            BinaryOp::ArithShiftRight => f.write_str("arith_shift_right"),
            BinaryOp::And => f.write_str("and"),
            BinaryOp::Or => f.write_str("or"),
            BinaryOp::Xor => f.write_str("xor"),
            BinaryOp::FloatAdd => f.write_str("float_add"),
            BinaryOp::FloatSub => f.write_str("float_sub"),
            BinaryOp::FloatMul => f.write_str("float_mul"),
            BinaryOp::FloatDiv => f.write_str("float_div"),
            BinaryOp::FloatRem => f.write_str("float_rem"),
        }
    }
}
impl Region {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        indent: usize,
        rvsdg_mod: &RVSDGMod,
    ) -> std::fmt::Result {
        apply_indent(f, indent)?;
        write!(f, "region (")?;

        for (i, param) in rvsdg_mod.values
            [self.params.start as usize..self.params.len as usize + self.params.start as usize]
            .iter()
            .enumerate()
        {
            param.fmt(f, rvsdg_mod)?;
            if i != self.params.len as usize - 1 {
                f.write_str(", ")?;
            }
        }
        f.write_str(") {\n")?;
        let new_indent = indent + 2;
        // dbg!(
        //     &self
        //         .nodes
        //         .iter()
        //         .map(|id| (id, rvsdg_mod.get(*id)))
        //         .collect::<Vec<_>>()
        // );
        for (i, id) in self.nodes.iter().enumerate() {
            apply_indent(f, new_indent)?;
            let val = rvsdg_mod.get(*id);
            // regions nodes contains the lambda which contain the region, this approach is wrong
            // but for now just skip the last node which will contain the region
            if i != self.nodes.len() - 1 {
                write!(f, "%{} = ", id.0)?;
                val.fmt(f, rvsdg_mod)?;
                f.write_char('\n')?;
            }
        }

        apply_indent(f, indent)?;
        f.write_str("}\n")?;

        Ok(())
    }
}

fn apply_indent(f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
    for _ in 0..indent {
        f.write_char(' ')?;
    }

    Ok(())
}
