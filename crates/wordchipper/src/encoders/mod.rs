//! # Token Encoders
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{TokenEncoder, TokenEncoderBuilder, TokenType, UnifiedTokenVocab};
//!
//! fn example<T: TokenType>(
//!     vocab: Arc<UnifiedTokenVocab<T>>,
//!     batch: &[&str],
//! ) -> Vec<Vec<T>> {
//!     let encoder = TokenEncoderBuilder::default(vocab.clone());
//!     encoder.try_encode_batch(batch).unwrap()
//! }
//! ```

mod encoder_builder;
mod encoder_options;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod token_encoder;
pub mod token_span_encoder;

#[doc(inline)]
pub use encoder_builder::*;
#[doc(inline)]
pub use encoder_options::*;
#[doc(inline)]
pub use token_encoder::*;
