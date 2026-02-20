//! # Time Utilities

/// Time an operation; return (duration, result).
#[cfg(feature = "std")]
pub fn timeit<F, R>(f: F) -> (std::time::Duration, R)
where
    F: FnOnce() -> R,
{
    let t0 = std::time::Instant::now();
    let ret = f();
    let t1 = std::time::Instant::now();
    (t1 - t0, ret)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "std")]
    fn test_timeit() {
        use std::time::Duration;
        let (dur, _) = super::timeit(|| {
            std::thread::sleep(Duration::from_millis(5));
            123
        });
        assert!(dur >= Duration::from_millis(5));
    }
}
