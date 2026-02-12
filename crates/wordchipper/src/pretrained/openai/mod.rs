//! Public `OpenAI` Patterns, Constants, and Models.

pub mod patterns;
pub mod resources;
pub mod specials;

#[cfg(feature = "std")]
mod loaders;

#[cfg(feature = "std")]
#[doc(inline)]
pub use loaders::*;
