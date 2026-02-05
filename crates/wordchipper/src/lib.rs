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
pub mod spanning;
pub mod types;
pub mod vocab;
