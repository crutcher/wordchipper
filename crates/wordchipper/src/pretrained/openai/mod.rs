//! Public `OpenAI` Patterns, Constants, and Models.

#[cfg(feature = "std")]
pub mod factories;
pub mod patterns;
mod provider;
pub mod resources;
pub mod spanning;
pub mod specials;

#[cfg(feature = "std")]
#[doc(inline)]
pub use factories::{OATokenizer, OATokenizerIter};
#[doc(inline)]
pub use patterns::*;
#[doc(inline)]
pub use provider::*;
#[doc(inline)]
pub use spanning::*;
