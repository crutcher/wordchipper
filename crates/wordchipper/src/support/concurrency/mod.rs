//! # Concurrency Utilities

#[cfg(feature = "rayon")]
pub mod rayon;
#[cfg(feature = "std")]
pub mod threads;

mod pool_toy;

#[doc(inline)]
pub use pool_toy::PoolToy;
