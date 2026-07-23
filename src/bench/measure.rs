//! Generic subprocess measurement for the benchmark harness: timed
//! execution with a kill-on-timeout guard (whole process group), a median
//! helper for the terminal glance, and the CPU governor check. Wall time
//! is a plain `Instant` around the subprocess -- no tracing
//! instrumentation, so the numbers cross-check against a shell `time`.

use std::fs;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// One completed (or killed) subprocess.
#[derive(Debug)]
pub struct CommandOutcome {
    /// `None` when the command was killed at the timeout.
    pub status: Option<ExitStatus>,
    pub stderr: String,
}

impl CommandOutcome {
    pub fn success(&self) -> bool {
        self.status.map(|s| s.success()).unwrap_or(false)
    }
}

/// A subprocess run that did not succeed: whether it was killed at the
/// timeout, and its stderr.
#[derive(Debug)]
pub struct SampleFailure {
    pub timed_out: bool,
    pub stderr: String,
}

/// Spawn `cmd` and wait up to `timeout`, killing it if it runs over.
/// stderr is redirected to a file and read back (peak-RSS and diagnostics
/// live there); stdout is discarded. A file, not a pipe, because polling a
/// pipe deadlocks once its buffer fills. The poll-and-kill idiom matches
/// difftest's `spawn_timed`; fold the two together when the shared bench
/// crate is extracted.
///
/// The child is put in its own process group so that on timeout the whole
/// tree can be killed, not just the child: the things we run spawn
/// grandchildren (our driver spawns clang/opt/cc; `/usr/bin/time` wraps
/// its target), and killing only the direct child would leave those
/// running, burning CPU into the next measurement and leaking processes.
pub fn run_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
    io_dir: &Path,
) -> std::io::Result<CommandOutcome> {
    // Unique per call: an io_dir can be shared by concurrent measurements
    // (the parallel pass runs a program's levels together), so a fixed
    // stderr filename would let them clobber each other's diagnostics. The
    // temp file is deleted when `stderr_file` drops at the end of the call.
    let stderr_file = tempfile::Builder::new()
        .prefix("bench.")
        .suffix(".stderr")
        .tempfile_in(io_dir)?;
    let stderr_path = stderr_file.path();
    cmd.stdout(Stdio::null())
        .stderr(fs::File::create(stderr_path)?)
        .stdin(Stdio::null())
        // New process group led by the child (pgid == child pid); its
        // descendants inherit it unless they change it themselves (compilers
        // and linkers do not).
        .process_group(0);

    let start = Instant::now();
    let mut child = cmd.spawn()?;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if start.elapsed() >= timeout {
            kill_process_group(&child);
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(2));
    };

    let mut stderr = String::new();
    fs::File::open(stderr_path)?.read_to_string(&mut stderr)?;
    Ok(CommandOutcome { status, stderr })
}

/// SIGKILL every process in the child's group. `child` was spawned with
/// `process_group(0)`, so its pgid equals its pid, and a negative pid to
/// `kill(2)` targets the whole group -- reaching the grandchildren a plain
/// `child.kill()` would miss.
///
/// The `unsafe` is only because `libc::kill` is an FFI function; the call
/// itself passes two integers and touches no memory, so it cannot cause
/// undefined behaviour -- the worst case is `kill` returning an error
/// (e.g. the group already exited), which we ignore. The `nix` crate
/// offers a safe `killpg` wrapper, but it is this same one-line syscall
/// inside nix's own `unsafe` block, so it would add a dependency without
/// removing any real risk.
fn kill_process_group(child: &Child) {
    let pgid = child.id() as i32;
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

/// Upper-middle median of a sample slice, `None` if empty. A glance
/// figure for the terminal table; the report recomputes proper medians
/// from the raw samples.
pub fn median<T: Copy + PartialOrd>(samples: &[T]) -> Option<T> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN samples"));
    Some(sorted[sorted.len() / 2])
}

/// The active CPU frequency governor, when readable. `performance` is the
/// one that keeps timing stable; anything else makes wall noisy.
pub fn cpu_governor() -> Option<String> {
    fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .ok()
        .map(|g| g.trim().to_string())
        .filter(|g| !g.is_empty())
}

/// A one-line warning when the governor will make timing noisy, or `None`
/// when the machine is configured for benchmarking. Linux-only by design
/// (the whole pipeline already is).
pub fn cpu_governor_warning() -> Option<String> {
    let governor = cpu_governor()?;
    (governor != "performance").then(|| {
        format!(
            "cpufreq governor is '{governor}', not 'performance'; runtimes will be noisy \
             (sudo cpupower frequency-set -g performance)"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A timed-out command must take its whole process tree with it, not
    /// just the immediate child. The shell backgrounds a long `sleep`
    /// (a grandchild of the spawned process), records its pid, then blocks
    /// on `wait`; after the timeout kill, that grandchild must be gone.
    #[test]
    fn timeout_kills_the_whole_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("grandchild.pid");
        let script = format!("sleep 300 & echo $! > {}; wait", pidfile.display());
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&script);

        let outcome = run_with_timeout(&mut cmd, Duration::from_millis(300), dir.path()).unwrap();
        assert!(outcome.status.is_none(), "command should have timed out");

        let pid: i32 = fs::read_to_string(&pidfile)
            .expect("grandchild pid was recorded")
            .trim()
            .parse()
            .unwrap();
        // Let the kernel reap the SIGKILLed grandchild.
        std::thread::sleep(Duration::from_millis(150));
        // kill(pid, 0) probes existence: 0 = still alive, -1/ESRCH = gone.
        let still_alive = unsafe { libc::kill(pid, 0) } == 0;
        assert!(!still_alive, "grandchild {pid} survived the group kill");
    }

    /// A command that finishes within the timeout reports its real status
    /// and is not treated as killed.
    #[test]
    fn fast_command_is_not_killed() {
        let dir = tempfile::tempdir().unwrap();
        let mut cmd = Command::new("true");
        let outcome = run_with_timeout(&mut cmd, Duration::from_secs(5), dir.path()).unwrap();
        assert!(outcome.success());
        assert!(outcome.status.is_some());
    }
}
