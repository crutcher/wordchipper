//! # Cross-Rust Version Compatibility

#[cfg(feature = "std")]
pub mod concurrency;
pub mod ranges;
pub mod regex;
pub mod resources;
pub mod slices;
pub mod strings;
pub mod timers;
pub mod traits;
