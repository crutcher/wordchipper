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
//!     decoder.try_decode_batch_to_strings(batch).unwrap()
//! }
//! ```

pub mod decode_context;
pub mod dictionary_decoder;
pub mod token_decoder;
pub mod utility;

#[doc(inline)]
pub use decode_context::TokenDecodeContext;
#[doc(inline)]
pub use dictionary_decoder::DictionaryDecoder;
#[doc(inline)]
pub use token_decoder::TokenDecoder;
