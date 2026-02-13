//! # Vocabulary IO
//!
//! ## Loading A Vocab
//!
//! ```rust,no_run
//! use wordchipper::{
//!     decoders::DefaultTokenDecoder,
//!     encoders::DefaultTokenEncoder,
//!     pretrained::openai::OA_O200K_BASE_PATTERN,
//!     spanning::TextSpanningConfig,
//!     vocab::{
//!         SpanMapVocab,
//!         SpanTokenMap,
//!         UnifiedTokenVocab,
//!         io::load_base64_unified_vocab_path,
//!     },
//! };
//!
//! fn example() -> anyhow::Result<(DefaultTokenEncoder<u32>, DefaultTokenDecoder<u32>)> {
//!     type T = u32;
//!     let vocab: UnifiedTokenVocab<T> = load_base64_unified_vocab_path(
//!         "vocab.tiktoken",
//!         TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN),
//!     )
//!     .expect("failed to load vocab");
//!
//!     let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::new(vocab.clone(), None);
//!     let decoder: DefaultTokenDecoder<T> = DefaultTokenDecoder::from_unified_vocab(vocab);
//!
//!     Ok((encoder, decoder))
//! }
//! ```

mod base64_vocab;

#[doc(inline)]
pub use base64_vocab::*;
