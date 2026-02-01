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
//! #### feature: ``default``
//!
//! * ``client``
//! * ``download``
//!
//!
//! The default feature does not enable "training".
//!
//! #### feature: ``client``
//!
//! * ``ahash``
//! * ``rayon``
//! * ``std``
//!
//! The default client is focused on loading vocabularies and running
//! high performance encoders / decoders.
//!
//! #### feature: ``download``
//!
//! * ``wordchipper-disk-cache``
//! * ``std``
//!
//! The download feature enables downloading vocabularies from the internet.
//!
//! #### feature: ``training``
//!
//! * ``compact_str``
//! * ``dary_heap``
//! * ``std``
//!
//! The training feature enables the training code.
//!
//! #### feature: ``std`` / ``no_std``
//!
//! The "std" feature enables the use of the `std` library;
//! and the "`no_std`" feature enables deps needed when "std" is not enabled.
//! (Negative feature deps are not stable yet.)
//!
//! Note: I am unsure if this is complete. It is tested CI, but I'm unsure
//! if I've fully covered it; and I haven't worked out a ``no_std`` deploy test yet.
//!
//! #### feature: ``ahash``
//!
//! This swaps all HashMap/HashSet implementations for ``ahash``; which is a performance
//! win on many/(most?) modern CPUs.
//!
//! This is done by the ``types::hash_types::CommonHash{*}`` type alias machinery.
//! See also the ``hashbrown`` dep used by ``no_std``.
//!
//! #### feature: ``rayon``
//!
//! This enables some parallelism wrappers using the ``rayon`` crate.
//!
//! TODO: I intend on providing a ``tokio`` based ``async`` parallelism mechanism
//! as well, to structure more direct ``regex find > encode span`` pipeline parallelism.
//!
//! #### feature: ``tracing``
//!
//! This enables a number of ``tracing`` instrumentation points.
//! This is only useful for timing tracing of the library itself.
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
//! use wordchipper::segmentation::{SegmentationConfig, TextSegmentor};
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
//! let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::init_with_factory(
//!     vocab.clone(), regex_pool_supplier);
//! let encoder = ParallelRayonEncoder::new(encoder);
//!
//! let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
//! let decoder = ParallelRayonDecoder::new(decoder);
//!```
#![warn(missing_docs, unused)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "rayon")]
pub mod rayon;

#[cfg(feature = "training")]
pub mod training;

#[cfg(feature = "download")]
#[doc(inline)]
pub use wordchipper_disk_cache as disk_cache;

pub mod decoders;
pub mod encoders;
pub mod regex;
pub mod segmentation;
pub mod types;
pub mod vocab;
