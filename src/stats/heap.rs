//! Rust-heap accounting: a global-allocator wrapper counting live and
//! peak bytes. Binaries that report memory install it:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: lang_rvsdg::stats::heap::CountingAllocator = CountingAllocator;
//! ```
//!
//! The library never installs it, so embedders keep their own allocator
//! choice; when it is not installed the counters read zero and reports
//! omit the numbers.
//!
//! Counting is OFF until [`enable`] is called: enabled, an alloc plus
//! its dealloc costs five relaxed atomic RMWs (live/peak plus the two
//! event counters), on the order of half a percent of a large compile
//! through the allocation-heavy llvm-ir parse -- an asymmetric handicap
//! in compile-time comparisons where the reference compiler carries no
//! equivalent. Disabled, every allocation pays one predictable branch
//! and the counters read zero.
//!
//! Scope: this sees exactly the Rust heap. LLVM's own C++ allocations
//! (context, output module, codegen) and the clang/opt frontend
//! subprocesses are invisible here -- the gap between peak RSS and
//! these counters is them. Vec::truncate keeps capacity, so a pass
//! that shrinks the graph does not lower live bytes; the census byte
//! budget reports the logical sizes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

#[derive(Debug)]
pub struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
// Signed, so enabling after startup is safe: deallocations of the few
// allocations that predate enabling (argument parsing) subtract from a
// counter that never saw them, which a signed value absorbs and the
// clamped readers hide. The skew is KBs against MB-scale measurements.
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static PEAK_BYTES: AtomicI64 = AtomicI64::new(0);
// Monotonic allocation-event counters (never decremented), the churn
// view live/peak cannot give: a region that allocates and frees scratch
// in a loop reads ~0 live but its full traffic here. `alloc_count` is
// the allocator-call-overhead proxy and, being near-deterministic for a
// given compile, a regression signal; `alloc_bytes` counts every byte
// ever handed out -- a realloc contributes its FULL new size (matching
// the copy traffic a grown vector causes), so growth-doubling shows up
// loudly. Unsigned is safe: nothing subtracts.
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// Turn counting on, normally right after CLI parsing decides the
/// numbers are wanted. One-way.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn record_alloc(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes as i64, Ordering::Relaxed) + bytes as i64;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

/// One allocator call handing out `bytes` (alloc, alloc_zeroed, or
/// realloc with its full new size).
fn record_event(bytes: usize) {
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOC_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

// SAFETY: delegates every operation verbatim to `System` and only
// OBSERVES sizes -- it never changes what is allocated, returned, or
// freed, so `System`'s upholding of the GlobalAlloc contract is
// inherited unchanged. The counter updates allocate nothing (atomics),
// so there is no reentrancy.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && enabled() {
            record_event(layout.size());
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && enabled() {
            record_event(layout.size());
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        if enabled() {
            LIVE_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && enabled() {
            record_event(new_size);
            if new_size >= layout.size() {
                record_alloc(new_size - layout.size());
            } else {
                LIVE_BYTES.fetch_sub((layout.size() - new_size) as i64, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

/// Bytes currently allocated on the Rust heap; zero when the counting
/// allocator is not installed or counting is not enabled.
pub fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed).max(0) as usize
}

/// Highest `live_bytes` ever observed in this process; zero when the
/// counting allocator is not installed or counting is not enabled.
pub fn peak_bytes() -> usize {
    PEAK_BYTES.load(Ordering::Relaxed).max(0) as usize
}

/// A point-in-time reading of the monotonic allocation counters. The
/// counters only ever grow, so the traffic of a measured region is the
/// difference of the snapshots taken around it.
#[derive(Debug, Clone, Copy)]
pub struct AllocSnapshot {
    /// Allocator calls handing out memory (alloc, alloc_zeroed, realloc).
    pub count: u64,
    /// Bytes handed out by those calls; a realloc contributes its full
    /// new size.
    pub bytes: u64,
}

impl AllocSnapshot {
    /// The traffic between `earlier` and this snapshot.
    pub fn since(&self, earlier: &AllocSnapshot) -> AllocSnapshot {
        AllocSnapshot {
            count: self.count - earlier.count,
            bytes: self.bytes - earlier.bytes,
        }
    }
}

/// The allocation counters now, or `None` while counting is off (a zero
/// reading would be indistinguishable from "measured, nothing
/// allocated"). Meaningful only in binaries that installed the
/// allocator; enabling without installing reads a permanent zero.
pub fn alloc_snapshot() -> Option<AllocSnapshot> {
    enabled().then(|| AllocSnapshot {
        count: ALLOC_COUNT.load(Ordering::Relaxed),
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
    })
}
