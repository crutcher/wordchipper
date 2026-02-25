//! Public `OpenAI` Patterns, Constants, and Models.

pub mod factories;
pub mod patterns;
mod provider;
pub mod resources;
pub mod spanning;
pub mod specials;

#[doc(inline)]
pub use factories::*;
#[doc(inline)]
pub use patterns::*;
#[doc(inline)]
pub use provider::*;
#[doc(inline)]
pub use spanning::*;
