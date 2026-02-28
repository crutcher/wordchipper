//! # [`SpanLexer`] mechanics for [`TextSpanner`](`crate::spanners::TextSpanner`) implementations.

pub mod logos;

pub mod accelerators;
mod lexer_builder;
mod lexer_spanner;
mod span_lexer;

#[doc(inline)]
pub use lexer_builder::*;
#[doc(inline)]
pub use lexer_spanner::*;
#[doc(inline)]
pub use span_lexer::*;
