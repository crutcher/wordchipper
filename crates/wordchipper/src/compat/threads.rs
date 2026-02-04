//! # Thread Utilities

use core::num::NonZeroU64;
use core::str::FromStr;
use std::num::NonZeroUsize;
use std::{env, thread};

/// Current Thread -> u64 Pool.
///
/// ``thread::current().id().as_u64()`` is unstable.
pub fn unstable_current_thread_id_hash() -> usize {
    // c/o `tiktoken`:
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread id counter
    // that works great for our use case of avoiding collisions in our array. Unfortunately,
    // it's private. However, there are only so many ways you can layout a u64, so just transmute
    // https://github.com/rust-lang/rust/issues/67939

    struct FakeThreadId(NonZeroU64);
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let val = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(val) as usize
}

/// Get the max parallelism available.
pub fn est_max_parallelism() -> usize {
    let default = || {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };

    #[cfg(feature = "rayon")]
    {
        match env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| usize::from_str(&s).ok())
        {
            Some(x @ 1..) => return x,
            Some(0) => return default(),
            _ => {}
        }

        // Support for deprecated `RAYON_RS_NUM_CPUS`.
        match env::var("RAYON_RS_NUM_CPUS")
            .ok()
            .and_then(|s| usize::from_str(&s).ok())
        {
            Some(x @ 1..) => x,
            _ => default(),
        }
    }

    #[cfg(not(feature = "rayon"))]
    default()
}

/// Resolve the max pool size.
///
/// ``min(max_pool, thread::available_parallelism() || MAX_POOL, env::var("RAYON_NUM_THREADS"))``
pub fn resolve_max_pool(max_pool: Option<NonZeroUsize>) -> usize {
    let sys_max = est_max_parallelism();

    let max_pool = max_pool.map(|x| x.get()).unwrap_or(sys_max);

    core::cmp::min(max_pool, sys_max)
}
