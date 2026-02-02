//! # Cross-Rust Version Compatibility

pub mod strings;

#[cfg(feature = "std")]
pub mod threads;
