//! # `wordchipper` LLM Tokenizer Suite
//!
//! This is a high-performance LLM tokenizer suite.
//!
//! `wordchipper` is compatible with `nanochat/rustbpe` and `tiktoken` tokenizers.
//!
//! See:
//! * [`encoders`] to encode text into tokens.
//! * [`decoders`] to decode tokens into text.
//! * [`training`] to train a [`vocab::UnifiedTokenVocab`].
//! * [`vocab`] to manage token vocabularies, vocab io, and pre-trained tokenizers.
//!
//! A number of pretrained public tokenizers are available through:
//! * [`vocab::public`]
//!
//! ## Crate Features
//!
#![doc = document_features::document_features!()]
//!
//! ## Loading Pretrained Tokenizers
//!
//! A number of pretrained tokenizers are available through:
//! * [`vocab::public`]
//!
//! ```rust,ignore
//! use wordchipper::decoders::{DictionaryDecoder, TokenDecoder};
//! use wordchipper::encoders::{DefaultTokenEncoder, TokenEncoder};
//! use wordchipper::rayon::{ParallelRayonDecoder, ParallelRayonEncoder};
//! use wordchipper::regex::{regex_pool_supplier, RegexWrapperPattern};
//! use wordchipper::spanner::{SegmentationConfig, TextSegmentor};
//! use wordchipper::vocab::public::openai::load_o200k_harmony_vocab;
//! use wordchipper::vocab::UnifiedTokenVocab;
//! use wordchipper::disk_cache::WordchipperDiskCache;
//!
//! type T = u32;
//!
//! let mut disk_cache = WordchipperDiskCache::default();
//! let vocab: UnifiedTokenVocab<T> =
//!     load_o200k_harmony_vocab(&mut disk_cache)?.into();
//!
//! let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::init(
//!     vocab.clone(), None);
//! let encoder = ParallelRayonEncoder::new(encoder);
//!
//! let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
//! let decoder = ParallelRayonDecoder::new(decoder);
//!```
#![warn(missing_docs, unused)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "training")]
pub mod training;

#[cfg(feature = "download")]
#[doc(inline)]
pub use wordchipper_disk_cache as disk_cache;

#[cfg(feature = "std")]
pub mod concurrency;

pub mod compat;
pub mod decoders;
pub mod encoders;
pub mod regex;
pub mod spanner;
pub mod types;
pub mod vocab;
