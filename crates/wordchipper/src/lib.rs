#![warn(missing_docs, unused)]
#![cfg_attr(not(feature = "std"), no_std)]
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
//! A number of pre-trained tokenizers are available through:
//! * [`pretrained`]
//!
//! ## Crate Features
#![doc = document_features::document_features!()]
//!
//! ## Loading Pretrained Models
//!
//! Loading a pre-trained model requires reading the vocabulary,
//! as well as configuring the spanning (regex and special words)
//! configuration.
//!
//! For a number of pretrained models, simplified constructors are
//! available to download, cache, and load the vocabulary.
//!
//! See: [`pretrained::openai::OATokenizer`]
#![cfg_attr(feature = "std", doc = "```rust,no_run")]
#![cfg_attr(not(feature = "std"), doc = "```ignore")]
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     decoders::{DefaultTokenDecoder, TokenDecoder},
//!     disk_cache::WordchipperDiskCache,
//!     encoders::{DefaultTokenEncoder, TokenEncoder},
//!     pretrained::openai::OATokenizer,
//!     vocab::UnifiedTokenVocab,
//! };
//!
//! fn example() -> anyhow::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
//!     let model = OATokenizer::O200kHarmony;
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: UnifiedTokenVocab<u32> = model.load(&mut disk_cache)?;
//!
//!     let encoder: Arc<DefaultTokenEncoder<u32>> =
//!         DefaultTokenEncoder::new(vocab.clone(), None).into();
//!     let decoder: Arc<DefaultTokenDecoder<u32>> =
//!         DefaultTokenDecoder::from_unified_vocab(vocab).into();
//!
//!     #[cfg(feature = "rayon")]
//!     use wordchipper::concurrency::rayon::*;
//!
//!     #[cfg(feature = "rayon")]
//!     let encoder = Arc::new(ParallelRayonEncoder::new(encoder));
//!
//!     #[cfg(feature = "rayon")]
//!     let decoder = Arc::new(ParallelRayonDecoder::new(decoder));
//!
//!     Ok((encoder, decoder))
//! }
//! ```

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
pub mod pretrained;
pub mod regex;
pub mod spanning;
pub mod types;
pub mod vocab;
