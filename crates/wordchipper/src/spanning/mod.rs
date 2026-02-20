//! # Text Segmentation
//!
//! This module exists to factor out text spanning scanning.
//!
//! [`TextSpanningConfig`] describes the declarative needs of a tokenizer:
//! * `pattern` - the word/span split pattern.
//! * `specials` - a map of `{ Vec<u8> -> T }` special tokens to handle out-of-band.
//!
//! Most users will want to use the [`TextSpannerBuilder`] to construct a [`TextSpanner`].

mod lexer_spanner;
#[cfg(feature = "logos")]
pub(crate) mod logos_lexer;
mod spanner_builder;
mod spanning_config;
mod text_spanner;

#[doc(inline)]
pub use lexer_spanner::*;
#[cfg(feature = "logos")]
#[doc(inline)]
pub use logos_lexer::{Cl100kLexer, O200kLexer};
#[doc(inline)]
pub use spanner_builder::*;
#[doc(inline)]
pub use spanning_config::*;
#[doc(inline)]
pub use text_spanner::*;
