#![warn(missing_docs, unused)]
#![cfg_attr(not(feature = "std"), no_std)]
//! # `wordchipper` LLM Tokenizer Suite
//!
//! This is a high-performance LLM tokenizer suite.
//!
//! ## Client Summary
//!
//! ### Core Client Types
//! * [`TokenType`] - the parameterized integer type used for tokens; choose from `{ u16, u32, u64 }`.
//! * [`UnifiedTokenVocab<T>`] - the unified vocabulary type.
//! * [`TokenEncoder<T>`] and [`TokenDecoder<T>`] - the encoder and decoder interfaces.
//!
//! ### Pre-Trained Models
//! * [`WordchipperDiskCache`](`disk_cache::WordchipperDiskCache`) - the disk cache for loading models.
//! * [`OATokenizer`](`pretrained::openai::OATokenizer`) - public pre-trained `OpenAI` tokenizers.
//!
//! ## `TokenType` and `WCHash`* Types
//!
//! `wordchipper` is parameterized over an abstract primitive integer [`TokenType`].
//! This permits vocabularies and tokenizers in the `{ u16, u32, u64 }` types.
//!
//! It is also feature-parameterized over the [`WCHashSet`] and [`WCHashMap`] types,
//! which are used to represent sets and maps of tokens.
//! These are provided for convenience and are not required for correctness.
//!
//!
//! ## Text Spanning
//!
//! Text spanning, splitting text into word and special token spans,
//! is defined by a [`spanning::TextSpanningConfig`];
//! and provided by implementors of [`spanning::TextSpanner`],
//! such as the [`spanning::RegexTextSpanner`].
//!
//! These interfaces can be used independently of the [`TokenEncoder`] and [`TokenDecoder`]
//! interfaces.
//!
//! ## Unified Vocabulary
//!
//! The core user-facing vocabulary type is [`UnifiedTokenVocab<T>`].
//!
//! Pre-trained vocabulary loaders return [`UnifiedTokenVocab<T>`] instances,
//! which can be converted between [`TokenType`]s via [`UnifiedTokenVocab::to_token_type`].
//!
//! Default [`TokenEncoder`] and [`TokenDecoder`] implementations
//! can be constructed directly using [`UnifiedTokenVocab::to_default_encoder`]
//! and [`UnifiedTokenVocab::to_default_decoder`].
//! [`UnifiedTokenVocab::to_encoder_builder`] and [`UnifiedTokenVocab::to_decoder_builder`]
//! return [`TokenEncoderBuilder`] and [`TokenDecoderBuilder`] instances,
//! which can be further configured with additional options.
//!
//! ## Loading and Saving Models
//!
//! Loading a pre-trained model requires reading in the vocabulary,
//! either as a [`vocab::SpanMapVocab`] or [`vocab::PairMapVocab`]
//! (either of which must have an attached [`vocab::ByteMapVocab`]);
//! and merging that with a [`spanning::TextSpanningConfig`]
//! to produce a [`UnifiedTokenVocab<T>`].
//!
//! A number of IO helpers are provided in [`vocab::io`].
//!
//! ## Loading Public Pre-trained Models
//!
//! For a number of pretrained models, simplified constructors are
//! available to download, cache, and load the vocabulary.
//!
//! Most users will want to use the [`get_model`] function, which will
//! return a [`UnifiedTokenVocab`] containing the vocabulary and
//! spanning configuration.
//!
//! There is also a [`list_models`] function which lists the available
//! pretrained models.
//!
//! See [`disk_cache::WordchipperDiskCache`] for details on the disk cache.
#![cfg_attr(feature = "std", doc = "```rust,no_run")]
#![cfg_attr(not(feature = "std"), doc = "```rust,ignore")]
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     get_model,
//!     TokenDecoder,
//!     TokenEncoder,
//!     UnifiedTokenVocab,
//!     disk_cache::WordchipperDiskCache,
//! };
//!
//! fn example() -> wordchipper::errors::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: UnifiedTokenVocab<u32> = get_model("openai/o200k_harmony", &mut disk_cache)?;
//!
//!     let encoder = vocab.to_default_encoder();
//!     let decoder = vocab.to_default_decoder();
//!
//!     Ok((encoder, decoder))
//! }
//! ```
//!
//! ## Training Models
//!
//! Training models is supported via the [`training`] module.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;

#[cfg(feature = "training")]
pub mod training;

#[cfg(feature = "download")]
#[doc(inline)]
pub use wordchipper_disk_cache as disk_cache;

#[cfg(feature = "std")]
pub mod concurrency;

pub mod errors;

pub mod compat;
pub mod decoders;
pub mod encoders;
pub mod pretrained;
pub mod regex;
pub mod resources;
pub mod spanning;
pub mod types;
pub mod vocab;

#[doc(inline)]
pub use decoders::{TokenDecoder, TokenDecoderBuilder};
#[doc(inline)]
pub use encoders::{TokenEncoder, TokenEncoderBuilder};
#[doc(inline)]
#[cfg(feature = "download")]
pub use pretrained::{get_model, list_models};
#[doc(inline)]
pub use types::*;
#[doc(inline)]
pub use vocab::{UnifiedTokenVocab, VocabIndex};
