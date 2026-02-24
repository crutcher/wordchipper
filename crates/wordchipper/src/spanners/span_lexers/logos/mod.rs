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
mod token_role;

#[doc(inline)]
pub use cl100k::Cl100kLexer;
#[doc(inline)]
pub use engine::for_each_classified_span;
#[doc(inline)]
pub use o200k::O200kLexer;
#[doc(inline)]
pub use token_role::{TokenRole, contraction_split};

/// Look up an accelerated word lexer for a known regex pattern.
///
/// Returns `Some(lexer)` when the pattern matches a pattern for which
/// a compile-time DFA lexer exists, `None` otherwise.
pub(crate) fn lookup_word_lexer(
    pattern: &crate::support::regex::RegexPattern
) -> Option<crate::alloc::sync::Arc<dyn super::SpanLexer>> {
    use crate::{
        alloc::sync::Arc,
        pretrained::openai::{OA_CL100K_BASE_PATTERN, OA_O200K_BASE_PATTERN},
    };

    let pat = pattern.as_str();
    if pat == OA_CL100K_BASE_PATTERN.as_str() {
        Some(Arc::new(Cl100kLexer))
    } else if pat == OA_O200K_BASE_PATTERN.as_str() {
        Some(Arc::new(O200kLexer))
    } else {
        None
    }
}
