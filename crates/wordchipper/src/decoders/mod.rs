//! # Token Decoders
//!
//! Decoder clients should use:
//!
//! * `DictionaryDecoder` - the fastest `TokenDecoder`.
//! * `ParallelRayonDecoder` - a batch parallelism wrapper around any `TokenDecoder`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use wordchipper::vocab::UnifiedTokenVocab;
//! use wordchipper::decoders::DictionaryDecoder;
//! use wordchipper::decoders::TokenDecoder;
//! use wordchipper::types::TokenType;
//!
//! fn example<T: TokenType>(
//!     vocab: UnifiedTokenVocab<T>,
//!     batch: &[Vec<T>],
//! ) -> Vec<String> {
//!     let decoder: DictionaryDecoder<T> = DictionaryDecoder::from_unified_vocab(vocab);
//!
//!     #[cfg(feature = "rayon")]
//!     let decoder = wordchipper::rayon::ParallelRayonDecoder::new(decoder);
//!
//!     let slices: Vec<&[T]> = batch.iter().map(|v| v.as_ref()).collect();
//!
//!     decoder.try_decode_batch_to_strings(&slices).unwrap().unwrap()
//! }
//! ```

pub mod dictionary_decoder;
pub mod token_decoder;
pub mod utility;

#[doc(inline)]
pub use dictionary_decoder::DictionaryDecoder;
#[doc(inline)]
pub use token_decoder::TokenDecoder;
