//! # Vocabulary
//!
//! This module provides the vocabulary and related io mechanisms.
//!
//! ## Byte Vocabulary
//!
//! Due to choices which exist in the community, we are forced to explicitly
//! map between byte values and token ranks. This is provided by:
//! * [`ByteMapVocab`].
//!
//! ## Unified Vocabulary
//!
//! The primary user-oriented vocabulary is [`UnifiedTokenVocab`], which contains:
//! * `spanning` - a [`crate::spanning::TextSpanningConfig`],
//! * `span_vocab` - a [`SpanMapVocab`] ``{ Vec<u8> -> T }`` vocabulary,
//! * `pair_vocab` - a [`PairMapVocab`] ``{ (T, T) -> T }`` vocabulary.
#[cfg(feature = "std")]
pub mod io;
pub mod utility;

mod byte_vocab;
mod pair_vocab;
mod span_vocab;
mod special_vocab;
mod token_vocab;
mod unified_vocab;
mod vocab_types;

#[doc(inline)]
pub use byte_vocab::ByteMapVocab;
#[doc(inline)]
pub use pair_vocab::PairMapVocab;
#[doc(inline)]
pub use span_vocab::SpanMapVocab;
#[doc(inline)]
pub use special_vocab::SpecialVocab;
#[doc(inline)]
pub use token_vocab::TokenVocab;
#[doc(inline)]
pub use unified_vocab::UnifiedTokenVocab;
#[doc(inline)]
pub use vocab_types::*;

/// Expected bytes/token ratio.
///
/// This is an observed bytes/token ratio, as a baseline
/// for scaling encode/decode buffers. Different languages
/// and encodings will see different ratios, and it
/// may be worth adjusting the ratio used by encoders/decoders
/// in production settings.
pub const DEFAULT_BYTE_PER_TOKEN_RATIO: f32 = 4.8;
