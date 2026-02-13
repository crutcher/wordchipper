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
//!     let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache)?;
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

pub mod openai;
