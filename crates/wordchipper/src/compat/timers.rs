//! # Time Utilities

use std::time::Duration;

/// Time an operation; return (duration, result).
pub fn timeit<F, R>(f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let t0 = std::time::Instant::now();
    let ret = f();
    let t1 = std::time::Instant::now();
    (t1 - t0, ret)
}
