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

pub mod byte_vocab;
pub mod pair_vocab;
pub mod size_hints;
pub mod span_vocab;
pub mod special_vocab;
pub mod token_vocab;
pub mod unified_vocab;
pub mod utility;
pub mod vocab_types;

#[doc(inline)]
pub use byte_vocab::ByteMapVocab;
#[doc(inline)]
pub use pair_vocab::PairMapVocab;
#[doc(inline)]
pub use span_vocab::SpanMapVocab;
#[doc(inline)]
pub use token_vocab::TokenVocab;
#[doc(inline)]
pub use unified_vocab::UnifiedTokenVocab;
#[doc(inline)]
pub use vocab_types::{
    ByteTokenArray, ByteTokenMap, PairTokenMap, SpanTokenMap, TokenByteMap, TokenPairMap,
    TokenSpanMap,
};
