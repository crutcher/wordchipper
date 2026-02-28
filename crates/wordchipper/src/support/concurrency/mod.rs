//! # Concurrency Utilities

#[cfg(feature = "parallel")]
pub mod rayon;
#[cfg(feature = "std")]
pub mod threads;

mod pool_toy;

#[doc(inline)]
pub use pool_toy::PoolToy;
