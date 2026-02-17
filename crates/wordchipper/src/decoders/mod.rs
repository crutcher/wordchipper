//! # Token Decoders
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{TokenDecoder, TokenDecoderBuilder, TokenType, UnifiedTokenVocab};
//!
//! fn example<T: TokenType>(
//!     vocab: UnifiedTokenVocab<T>,
//!     batch: &[Vec<T>],
//! ) -> Vec<String> {
//!     let decoder = TokenDecoderBuilder::default(vocab);
//!
//!     let slices: Vec<&[T]> = batch.iter().map(|v| v.as_ref()).collect();
//!
//!     decoder
//!         .try_decode_batch_to_strings(&slices)
//!         .unwrap()
//!         .unwrap()
//! }
//! ```

pub mod utility;

mod decode_results;
mod decoder_builder;
mod token_decoder;
mod token_dict_decoder;

#[doc(inline)]
pub use decode_results::*;
#[doc(inline)]
pub use decoder_builder::*;
#[doc(inline)]
pub use token_decoder::*;
#[doc(inline)]
pub use token_dict_decoder::*;
