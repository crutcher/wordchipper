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
//!     let encoder = wordchipper::concurrency::rayon::ParallelRayonEncoder::new(encoder);
//!
//!     encoder.try_encode_batch(batch).unwrap()
//! }
//! ```

mod token_encoder;
#[doc(inline)]
pub use token_encoder::TokenEncoder;

pub mod span_encoders;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

/// The default `TokenEncoder` implementation.
///
/// ## Style Hints
///
/// When there is no local ambiguity with other encoders,
/// prefer `decoder` for instance names.
pub type DefaultTokenEncoder<T> = span_encoders::CompoundSpanVocabEncoder<T>;
