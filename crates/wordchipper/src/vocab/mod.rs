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
//! [`UnifiedTokenVocab<T>`] is the primary vocabulary type for end users.
//! It unifies several component vocabularies into a coherent interface:
//! * [`ByteMapVocab`] - bidirectional byte⟷token mapping
//! * [`PairMapVocab`] - BPE merge pair mapping: `(T, T) → T`
//! * [`SpanMapVocab`] - span dictionary mapping: `Vec<u8> → T`
//! * [`crate::spanning::TextSpanningConfig`] - text spanning configuration
//!   that defines how text is split into spans for encoding, including
//!   special token words
//!
//! Pre-trained vocabulary loaders return [`UnifiedTokenVocab<T>`] instances,
//! which can be converted between [`crate::TokenType`]s via [`UnifiedTokenVocab::to_token_type`].
//!
//! Default [`crate::TokenEncoder`] and [`crate::TokenDecoder`] implementations
//! can be constructed directly using [`UnifiedTokenVocab::to_default_encoder`]
//! and [`UnifiedTokenVocab::to_default_decoder`].
//! [`UnifiedTokenVocab::to_encoder_builder`] and
//! [`UnifiedTokenVocab::to_decoder_builder`] return [`crate::TokenEncoderBuilder`] and
//! [`crate::TokenDecoderBuilder`] instances, which can be further configured
//! with additional options.
//!
//! ## Loading and Saving Models
//!
//! Loading a pre-trained model requires reading in the vocabulary,
//! either as a [`SpanMapVocab`] or [`PairMapVocab`]
//! (either of which must have an attached [`ByteMapVocab`]);
//! and merging that with a [`crate::spanning::TextSpanningConfig`]
//! to produce a [`UnifiedTokenVocab<T>`].
//!
//! A number of IO helpers are provided in [`io`].
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
pub use byte_vocab::*;
#[doc(inline)]
pub use pair_vocab::*;
#[doc(inline)]
pub use span_vocab::*;
#[doc(inline)]
pub use special_vocab::*;
#[doc(inline)]
pub use token_vocab::*;
#[doc(inline)]
pub use unified_vocab::*;
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
