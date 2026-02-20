//! # [`SpanLexer`] mechanics for [`TextSpanner`](`crate::spanners::TextSpanner`) implementations.

#[cfg(feature = "logos")]
pub mod logos;

mod lexer_spanner;
mod regex_lexer_builder;
mod span_lexer;

#[doc(inline)]
pub use lexer_spanner::*;
#[doc(inline)]
pub use regex_lexer_builder::*;
#[doc(inline)]
pub use span_lexer::*;
