//! # Token Encoders
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     TokenEncoder,
//!     TokenEncoderBuilder,
//!     TokenType,
//!     UnifiedTokenVocab,
//!     spanning::RegexTextSpanner,
//! };
//!
//! fn example<T: TokenType>(
//!     vocab: UnifiedTokenVocab<T>,
//!     batch: &[&str],
//! ) -> Vec<Vec<T>> {
//!     let encoder = TokenEncoderBuilder::new(vocab.clone()).init();
//!
//!     encoder.try_encode_batch(batch).unwrap()
//! }
//! ```

mod encoder_builder;
pub mod span_encoders;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod token_encoder;

#[doc(inline)]
pub use encoder_builder::*;
#[doc(inline)]
pub use token_encoder::*;
