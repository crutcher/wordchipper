//! # Text Segmentation
//!
//! This module exists to factor out text spanner scanning.
//!
//! [`TextSpanConfig`] describes the declarative needs of a tokenizer:
//! * `pattern` - the word/span split pattern.
//! * `specials` - a map of `{ Vec<u8> -> T }` special tokens to handle out-of-band.
//!
//! [`TextSpanner`] implements the run-time management of spanner,
//! as well as any per-thread regex pooling.

mod spanner_config;
mod text_spanner;

#[doc(inline)]
pub use spanner_config::TextSpanConfig;
#[doc(inline)]
pub use text_spanner::{SpanRef, TextSpanner};
