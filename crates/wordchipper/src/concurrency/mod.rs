//! # Concurrency Utilities

#[cfg(feature = "experimental")]
pub mod lru_pool_toy;

pub mod pool_toy;
#[cfg(feature = "rayon")]
pub mod rayon;
#[cfg(feature = "std")]
pub mod threads;

pub use pool_toy::PoolToy;
