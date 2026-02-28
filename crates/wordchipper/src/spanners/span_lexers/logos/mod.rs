//! # Logos-based lexers
//!
//! Composable building blocks for compile-time DFA word scanners using
//! the [`logos`](https://docs.rs/logos) crate.
//!
//! ## Building a custom lexer
//!
//! 1. Define a `#[derive(Logos)]` enum with your token patterns.
//! 2. Map each variant to a [`TokenRole`] (via a `role()` method or similar).
//! 3. Implement [`SpanLexer`](super::SpanLexer) by feeding the token stream
//!    to [`for_each_classified_span`].
//!
//! See [`Cl100kLexer`] and [`O200kLexer`] for reference implementations.

mod cl100k;
mod engine;
mod o200k;
mod r50k;
mod token_role;

#[doc(inline)]
pub use cl100k::Cl100kLexer;
#[doc(inline)]
pub use engine::for_each_classified_span;
#[doc(inline)]
pub use o200k::O200kLexer;
#[doc(inline)]
pub use r50k::R50kLexer;
#[doc(inline)]
pub use token_role::{TokenRole, contraction_split};
