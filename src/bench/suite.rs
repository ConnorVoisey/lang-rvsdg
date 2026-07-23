//! Benchmark suites (presets): corpus knowledge in code, not a shell
//! script. Each preset turns a path into the [`Program`]s it contains
//! plus the frontend flags those programs need, and carries the
//! iteration counts and runtime capability that program mix warrants.
//!
//! This replaces both `graph_stats_corpus.sh` (env-var paths in a shell
//! script) and the standalone `polybench` binary's kernel discovery: the
//! knowledge of what a suite IS lives here, versioned with the code that
//! reads it. Presets compose -- one benchmark invocation can pull in
//! several suites and write a single run record spanning all of them.
//!
//! Only compile-time is measured today; the `runtime_capable` flag and
//! the polybench harness knowledge are recorded now so adding runtime
//! later populates fields rather than reshaping this.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{WrapErr, eyre};

use crate::bench::compile_time::Program;

/// A named set of programs and how they behave under measurement.
/// Iteration counts are a run-level choice (see the `compile_bench`
/// binary), uniform across suites until a first full run calibrates
/// per-suite counts.
#[derive(Debug)]
pub struct Suite {
    pub programs: Vec<Program>,
    /// Whether the produced binaries can be executed and timed. Reserved
    /// for the deferred runtime tier; false for compile-only suites.
    pub runtime_capable: bool,
}

/// PolyBench problem-size selection (`-D <SIZE>_DATASET`). Compile time
/// is dataset-independent (the loop bounds are constants that do not
/// change the IR shape), but the define must still be set or the kernel
/// fails to preprocess; the smallest is used.
#[derive(Debug, Clone, Copy)]
pub enum Dataset {
    Mini,
    Small,
    Medium,
    Large,
    ExtraLarge,
}

impl Dataset {
    pub fn define(self) -> &'static str {
        match self {
            Dataset::Mini => "MINI_DATASET",
            Dataset::Small => "SMALL_DATASET",
            Dataset::Medium => "MEDIUM_DATASET",
            Dataset::Large => "LARGE_DATASET",
            Dataset::ExtraLarge => "EXTRALARGE_DATASET",
        }
    }
}

/// The PolyBench suite: every kernel `utilities/benchmark_list` names,
/// each as one program compiled with the harness include path and a
/// dataset define. The harness translation unit
/// (`utilities/polybench.c`) is a link-time dependency, not a
/// compile-time one, so it is not needed until runtime lands.
pub fn polybench(suite_dir: &Path) -> color_eyre::Result<Suite> {
    let utilities = suite_dir.join("utilities");
    let list_path = utilities.join("benchmark_list");
    let list = std::fs::read_to_string(&list_path).wrap_err_with(|| {
        format!(
            "reading {} (is the path a PolyBench 4.2.1 checkout?)",
            list_path.display()
        )
    })?;

    let utilities_flag = utilities.to_string_lossy().to_string();
    let mut programs = Vec::new();
    for line in list.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let relative = line.trim_start_matches("./");
        let source = suite_dir.join(relative);
        if !source.exists() {
            return Err(eyre!(
                "benchmark_list names {relative} but it does not exist"
            ));
        }
        let dir = source
            .parent()
            .ok_or_else(|| eyre!("benchmark_list entry {relative} has no directory"))?;
        let name = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| eyre!("benchmark_list entry {relative} has no UTF-8 stem"))?
            .to_string();
        programs.push(Program {
            name,
            input: source.clone(),
            includes: vec![utilities_flag.clone(), dir.to_string_lossy().to_string()],
            defines: vec![Dataset::Mini.define().to_string()],
        });
    }
    if programs.is_empty() {
        return Err(eyre!("{} lists no benchmarks", list_path.display()));
    }
    Ok(Suite {
        programs,
        runtime_capable: true,
    })
}

/// The sqlite amalgamation: one large translation unit, compile-only.
/// The path may be the amalgamation directory or the `sqlite3.c` file.
pub fn sqlite(path: &Path) -> color_eyre::Result<Suite> {
    let source = if path.is_dir() {
        path.join("sqlite3.c")
    } else {
        path.to_path_buf()
    };
    require(&source)?;
    Ok(single("sqlite3", source, Vec::new(), Vec::new()))
}

/// The Lua single-file build (`onelua.c`), compile-only.
pub fn lua(file: &Path) -> color_eyre::Result<Suite> {
    require(file)?;
    Ok(single(
        "lua",
        file.to_path_buf(),
        Vec::new(),
        vec![
            "LUA_USE_LINUX".to_string(),
            "LUA_USE_JUMPTABLE=0".to_string(),
        ],
    ))
}

/// The http-server unity build, compile-only. Its link-time flags
/// (`-lm -luring -lpq`) are runtime concerns; compilation needs only the
/// feature defines.
pub fn unity(file: &Path) -> color_eyre::Result<Suite> {
    require(file)?;
    Ok(single(
        "unity",
        file.to_path_buf(),
        Vec::new(),
        vec![
            "_GNU_SOURCE".to_string(),
            "MCO_USE_VMEM_ALLOCATOR".to_string(),
        ],
    ))
}

/// An ad-hoc single input with caller-supplied include/define flags.
pub fn program(
    file: &Path,
    includes: Vec<String>,
    defines: Vec<String>,
) -> color_eyre::Result<Suite> {
    require(file)?;
    let name = file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("input")
        .to_string();
    Ok(single(&name, file.to_path_buf(), includes, defines))
}

fn require(path: &Path) -> color_eyre::Result<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(eyre!("{} does not exist", path.display()))
    }
}

fn single(name: &str, input: PathBuf, includes: Vec<String>, defines: Vec<String>) -> Suite {
    Suite {
        programs: vec![Program {
            name: name.to_string(),
            input,
            includes,
            defines,
        }],
        runtime_capable: false,
    }
}
