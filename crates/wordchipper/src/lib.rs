#![no_std]
#![warn(missing_docs, unused)]
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
//! ## Unified Vocabulary
//!
//! The core user-facing vocabulary type is [`UnifiedTokenVocab<T>`].
//!
//! Pre-trained vocabulary loaders return [`UnifiedTokenVocab<T>`] instances,
//! which can be converted between [`TokenType`]s via [`UnifiedTokenVocab::to_token_type`].
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
//!     Tokenizer,
//!     TokenizerBuilder,
//!     UnifiedTokenVocab,
//!     WCResult,
//!     disk_cache::WordchipperDiskCache,
//! };
//!
//! fn example() -> WCResult< Arc<Tokenizer<u32>> > {
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: Arc<UnifiedTokenVocab<u32>> =
//!         get_model("openai/o200k_harmony", &mut disk_cache)?.into();
//!
//!     Ok(TokenizerBuilder::default(vocab.clone()))
//! }
//! ```
//! 
//! ## Training Models
//!
//! Training models is supported via the [`training`] module.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

#[cfg(feature = "std")]
extern crate std;

#[cfg_attr(feature = "std", macro_use)]
extern crate alloc;

/// Re-exports of common `alloc` types that are normally in the std prelude.
///
/// Modules that use `Vec`, `String`, `Box`, or `ToString` should add:
/// ```ignore
/// use crate::prelude::*;
/// ```
#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use alloc::{
        boxed::Box,
        string::{String, ToString},
        vec::Vec,
    };
}

#[cfg(feature = "training")]
pub mod training;
#[cfg(feature = "download")]
#[doc(inline)]
pub use wordchipper_disk_cache as disk_cache;
pub mod decoders;
pub mod encoders;
pub mod pretrained;
pub mod spanning;
pub mod support;
pub mod vocab;

mod errors;
mod tokenizer;
mod types;

#[doc(inline)]
pub use decoders::{TokenDecoder, TokenDecoderBuilder};
#[doc(inline)]
pub use encoders::{TokenEncoder, TokenEncoderBuilder};
#[doc(inline)]
pub use errors::*;
#[doc(inline)]
#[cfg(feature = "download")]
pub use pretrained::{get_model, list_models};
#[doc(inline)]
pub use tokenizer::*;
#[doc(inline)]
pub use types::*;
#[doc(inline)]
pub use vocab::{UnifiedTokenVocab, VocabIndex};
