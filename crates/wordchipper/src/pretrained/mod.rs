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
//! return a [`UnifiedTokenVocab<u32>`](`crate::UnifiedTokenVocab`)
//! containing the vocabulary and spanning configuration.
//!
//! There is also a [`list_models`] function which lists the available
//! pretrained models.
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     TokenDecoder,
//!     TokenDecoderBuilder,
//!     TokenEncoder,
//!     TokenEncoderBuilder,
//!     disk_cache::WordchipperDiskCache,
//!     get_model,
//!     vocab::UnifiedTokenVocab,
//! };
//!
//! fn example() -> anyhow::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: UnifiedTokenVocab<u32> = get_model("openai/o200k_harmony", &mut disk_cache)?;
//!
//!     let encoder: Arc<dyn TokenEncoder<u32>> = vocab.to_default_encoder();
//!     let decoder: Arc<dyn TokenDecoder<u32>> = vocab.to_default_decoder();
//!
//!     Ok((encoder, decoder))
//! }
//! ```

#[cfg(feature = "download")]
mod load_by_name;
pub mod openai;

#[doc(inline)]
#[cfg(feature = "download")]
pub use load_by_name::*;
