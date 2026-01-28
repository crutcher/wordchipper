//! # Token Encoders
//!
//! Encoder clients should use:
//!
//! * `MergeHeapVocabEncoder` - the current default (only?) `TokenEncoder`.
//! * `ParallelRayonEncoder` - a batch parallelism wrapper around any `TokenEncoder`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use wordchipper::vocab::UnifiedTokenVocab;
//! use wordchipper::encoders::MergeHeapVocabEncoder;
//! use wordchipper::encoders::TokenEncoder;
//! use wordchipper::types::TokenType;
//! use std::sync::Arc;
//!
//! fn example<T: TokenType>(
//!     vocab: Arc<UnifiedTokenVocab<T>>,
//!     batch: &[String],
//! ) -> Vec<Vec<T>> {
//!     let encoder: MergeHeapVocabEncoder<T> = MergeHeapVocabEncoder::init(vocab);
//!
//!     #[cfg(feature = "rayon")]
//!     let encoder = wordchipper::rayon::ParallelRayonEncoder::new(encoder);
//!
//!     encoder.try_encode_batch(batch).unwrap()
//! }
//! ```

pub mod merge_heap_encoder;
pub mod token_encoder;

#[doc(inline)]
pub use merge_heap_encoder::MergeHeapVocabEncoder;
#[doc(inline)]
pub use token_encoder::TokenEncoder;
