//! # Text Segmentation
//!
//! This module exists to factor out text segmentation scanning.
//!
//! [`SegmentationConfig`] describes the declarative needs of a tokenizer:
//! * `pattern` - the word/span split pattern.
//! * `specials` - a map of `{ Vec<u8> -> T }` special tokens to handle out-of-band.
//!
//! [`TextSegmentor`] implements the run-time management of segmentation,
//! as well as any per-thread regex pooling.

pub mod segmentation_config;
pub mod text_segmentor;

#[doc(inline)]
pub use segmentation_config::SegmentationConfig;
#[doc(inline)]
pub use text_segmentor::{SpanRef, TextSegmentor};
