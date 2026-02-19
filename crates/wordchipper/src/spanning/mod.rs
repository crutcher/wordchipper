//! # Text Segmentation
//!
//! This module exists to factor out text spanning scanning.
//!
//! [`TextSpanningConfig`] describes the declarative needs of a tokenizer:
//! * `pattern` - the word/span split pattern.
//! * `specials` - a map of `{ Vec<u8> -> T }` special tokens to handle out-of-band.
//!
//! [`RegexTextSpanner`] implements the run-time management of spanning,
//! as well as any per-thread regex pooling.

pub mod logos_text_spanner;
mod regex_text_spanner;
mod spanning_config;
mod text_spanner;

#[doc(inline)]
pub use logos_text_spanner::{LogosTextSpanner, LogosVariant};
#[doc(inline)]
pub use regex_text_spanner::*;
#[doc(inline)]
pub use spanning_config::*;
#[doc(inline)]
pub use text_spanner::*;
