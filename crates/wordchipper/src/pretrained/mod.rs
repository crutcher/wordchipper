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
//! use wordchipper::{
//!     decoders::DefaultTokenDecoder,
//!     disk_cache::WordchipperDiskCache,
//!     encoders::DefaultTokenEncoder,
//!     pretrained::openai::OATokenizer,
//!     vocab::UnifiedTokenVocab,
//! };
//!
//! fn example() -> anyhow::Result<(DefaultTokenEncoder<u32>, DefaultTokenDecoder<u32>)> {
//!     let model = OATokenizer::O200kHarmony;
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let vocab: UnifiedTokenVocab<u32> = model.load(&mut disk_cache)?;
//!
//!     let encoder: DefaultTokenEncoder<u32> = DefaultTokenEncoder::new(vocab.clone(), None);
//!     let decoder: DefaultTokenDecoder<u32> = DefaultTokenDecoder::from_unified_vocab(vocab);
//!
//!     Ok((encoder, decoder))
//! }
//! ```

pub mod openai;
