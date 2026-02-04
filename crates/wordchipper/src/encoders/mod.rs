//! # Token Encoders
//!
//! Encoder clients should use:
//!
//! * `DefaultTokenEncoder` - the current default (only?) `TokenEncoder`.
//! * `ParallelRayonEncoder` - a batch parallelism wrapper around any `TokenEncoder`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use wordchipper::vocab::UnifiedTokenVocab;
//! use wordchipper::encoders::DefaultTokenEncoder;
//! use wordchipper::encoders::TokenEncoder;
//! use wordchipper::types::TokenType;
//!
//! fn example<T: TokenType>(
//!     vocab: UnifiedTokenVocab<T>,
//!     batch: &[&str],
//! ) -> Vec<Vec<T>> {
//!     let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::init(vocab, None);
//!
//!     #[cfg(feature = "rayon")]
//!     let encoder = wordchipper::rayon::ParallelRayonEncoder::new(encoder);
//!
//!     encoder.try_encode_batch(batch).unwrap()
//! }
//! ```

#[cfg(test)]
pub mod test_utils;

#[doc(inline)]
pub use token_encoder::TokenEncoder;

pub mod merge_heap_encoder;
pub mod merge_scan_encoder;
pub mod span_encoder;
pub mod token_encoder;

/// The default `TokenEncoder` implementation.
pub type DefaultTokenEncoder<T> = merge_heap_encoder::MergeHeapVocabEncoder<T>;
