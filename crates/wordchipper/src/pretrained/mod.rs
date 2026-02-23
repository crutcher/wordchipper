//! # Public Vocabulary Information
//!
//! Common Stats, Public Patterns, and Pretrained Model Sources
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
//! Most users will want to use the [`get_model`] function, which will
//! return a [`Arc<UnifiedTokenVocab<u32>>`](`crate::UnifiedTokenVocab`)
//! containing the vocabulary and spanning configuration.
//!
//! There is also a [`list_models`] function which lists the available
//! pretrained models.
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     UnifiedTokenVocab,
//!     Tokenizer,
//!     TokenizerOptions,
//!     disk_cache::WordchipperDiskCache,
//!     get_model,
//! };
//!
//! fn example() -> wordchipper::WCResult< Arc<Tokenizer<u32>> > {
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: Arc<UnifiedTokenVocab<u32>> =
//!         get_model("openai/o200k_harmony", &mut disk_cache)?.into();
//!
//!     let tokenizer = TokenizerOptions::default().build(vocab);
//!
//!     Ok(tokenizer)
//! }

#[cfg(feature = "download")]
mod load_by_name;
pub mod openai;

#[doc(inline)]
#[cfg(feature = "download")]
pub use load_by_name::*;
