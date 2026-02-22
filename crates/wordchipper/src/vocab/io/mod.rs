//! # Vocabulary IO
//!
//! ## Loading A Vocab
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     TokenDecoder,
//!     TokenDecoderBuilder,
//!     TokenEncoder,
//!     TokenEncoderBuilder,
//!     UnifiedTokenVocab,
//!     pretrained::openai::OA_O200K_BASE_PATTERN,
//!     spanning::TextSpanningConfig,
//!     vocab::io::load_base64_unified_vocab_path,
//! };
//!
//! fn example() -> wordchipper::WCResult<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)>
//! {
//!     let vocab: Arc<UnifiedTokenVocab<u32>> = load_base64_unified_vocab_path(
//!         "vocab.tiktoken",
//!         TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN),
//!     )
//!     .expect("failed to load vocab")
//!     .into();
//!
//!     let encoder = TokenEncoderBuilder::default(vocab.clone());
//!     let decoder = TokenDecoderBuilder::default(vocab.clone());
//!
//!     Ok((encoder, decoder))
//! }
//! ```

mod base64_vocab;

#[doc(inline)]
pub use base64_vocab::*;
