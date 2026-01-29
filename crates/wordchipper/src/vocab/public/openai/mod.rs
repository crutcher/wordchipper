//! Public `OpenAI` Patterns, Constants, and Models.

pub mod patterns;
pub mod resources;
pub mod specials;

#[cfg(feature = "download")]
mod loaders;
#[cfg(feature = "download")]
#[doc(inline)]
pub use loaders::*;
