//! # Token Decoders
//!
//! Decoder clients should use:
//!
//! * `TokenDictDecoder` - the fastest `TokenDecoder`.
//! * `ParallelRayonDecoder` - a batch parallelism wrapper around any `TokenDecoder`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use wordchipper::vocab::UnifiedTokenVocab;
//! use wordchipper::decoders::TokenDictDecoder;
//! use wordchipper::decoders::TokenDecoder;
//! use wordchipper::types::TokenType;
//!
//! fn example<T: TokenType>(
//!     vocab: UnifiedTokenVocab<T>,
//!     batch: &[Vec<T>],
//! ) -> Vec<String> {
//!     let decoder: TokenDictDecoder<T> = TokenDictDecoder::from_unified_vocab(vocab);
//!
//!     #[cfg(feature = "rayon")]
//!     let decoder = wordchipper::concurrency::rayon::ParallelRayonDecoder::new(decoder);
//!
//!     let slices: Vec<&[T]> = batch.iter().map(|v| v.as_ref()).collect();
//!
//!     decoder.try_decode_batch_to_strings(&slices).unwrap().unwrap()
//! }
//! ```

pub mod utility;

mod decode_results;
mod token_decoder;
mod token_dict_decoder;

#[doc(inline)]
pub use decode_results::{BatchDecodeResult, DecodeResult};
#[doc(inline)]
pub use token_decoder::TokenDecoder;
#[doc(inline)]
pub use token_dict_decoder::TokenDictDecoder;
