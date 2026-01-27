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
//! * `segmentation` - a [`crate::segmentation::SegmentationConfig`],
//! * `span_vocab` - a [`SpanMapVocab`] ``{ Vec<u8> -> T }`` vocabulary,
//! * `pair_vocab` - a [`PairMapVocab`] ``{ (T, T) -> T }`` vocabulary.
//!
//! ## Public Pretrained
//!
//! Metadata and load support for a number of public pre-trained tokenizers
//! exists in [`public`].

#[cfg(feature = "std")]
pub mod io;

pub mod byte_vocab;
pub mod pair_vocab;
pub mod public;
pub mod span_vocab;
pub mod special_vocab;
pub mod token_vocab;
pub mod unified_vocab;
pub mod utility;

pub use byte_vocab::ByteMapVocab;
pub use pair_vocab::PairMapVocab;
pub use span_vocab::SpanMapVocab;
pub use token_vocab::TokenVocab;
pub use unified_vocab::UnifiedTokenVocab;
