//! # Thread Utilities

use core::num::NonZeroU64;
use std::thread;

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
