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
//! See: [`openai::OATokenizer`] for example options.
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
//!     pretrained::openai::OATokenizer,
//!     vocab::UnifiedTokenVocab,
//! };
//!
//! fn example() -> anyhow::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
//!     let model = OATokenizer::O200kHarmony;
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache)?;
//!
//!     let encoder: Arc<dyn TokenEncoder<u32>> = TokenEncoderBuilder::new(vocab.clone()).init();
//!     let decoder: Arc<dyn TokenDecoder<u32>> = TokenDecoderBuilder::new(vocab.clone()).init();
//!
//!     Ok((encoder, decoder))
//! }
//! ```

pub mod openai;
