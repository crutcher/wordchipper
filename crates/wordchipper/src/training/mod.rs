//! # Vocabulary Training
//!
//! Support for training token vocabularies.
//!
//! Training requires:
//! * [`crate::vocab::ByteMapVocab`] - a choice of ``{ u8 -> T }` byte mappings.
//!   * The default is ``T::from(u8)``.
//! * [`crate::spanning::TextSpanningConfig`] - a text splitting config.
//!   * This can be built from just a regex pattern.
//!   * Special tokens can be overlaid on a pre-trained vocabulary.
//!
//! ## Training Example
//!
//! See `examples/tokenizer_trainer`.
//!
//! This is a code snippet overview of training.
//!
//! Expect training to take ~1s/10MB of input; and to be slowed
//! primarily by how well the stream logic of loading the training
//! samples is parallelized.
//!
//! Note: currently, training has limited logging and no progress reporting.
//!
//! A common training binary is probably a good idea; and much of the messiness
//! of supporting many different training data sources could be hidden in
//! the isolated deps of such a tool.
//!
//! Consider the following, to train a tokenizer and export it a "*.tiktoken" file.
//!
//! - The iterator stream for samples may be quite large.
//! - Training a `nanochat` equivalent tokenizer takes ~80 CPU minutes.
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     TokenDecoderBuilder,
//!     TokenEncoderBuilder,
//!     UnifiedTokenVocab,
//!     pretrained::openai::OA_CL100K_BASE_PATTERN,
//!     training::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions},
//!     vocab::{ByteMapVocab, io::save_base64_span_map_path},
//! };
//!
//! fn example<I, S>(
//!     vocab_size: usize,
//!     batches: I,
//!     vocab_save_path: Option<String>,
//! ) where
//!     I: IntoIterator,
//!     I::Item: AsRef<[S]>,
//!     S: AsRef<str>,
//! {
//!     // We can pick any unsigned integer type > vocab_size;
//!     // See [`wordchipper::TokenType`].
//!     type T = u32;
//!     type K = String;
//!     type C = u64;
//!
//!     let options = BinaryPairVocabTrainerOptions::new(OA_CL100K_BASE_PATTERN, vocab_size);
//!
//!     let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();
//!
//!     for batch in batches {
//!         // The trainer has no parallelism.
//!         // The perceived benefits of parallelism in the trainer
//!         // are insignificant if the IO for the sample source is
//!         // fed by another thread.
//!         trainer.update_from_samples(batch.as_ref());
//!     }
//!
//!     let byte_vocab: ByteMapVocab<T> = Default::default();
//!
//!     let vocab: Arc<UnifiedTokenVocab<T>> = trainer
//!         .train(byte_vocab.clone())
//!         .expect("training failed")
//!         .into();
//!
//!     if let Some(path) = vocab_save_path {
//!         save_base64_span_map_path(&vocab.span_vocab().span_map(), &path)
//!             .expect("failed to save vocab");
//!         println!("- tiktoken vocab: {path:?}");
//!     }
//!
//!     let encoder = TokenEncoderBuilder::default(vocab.clone());
//!     let decoder = TokenDecoderBuilder::default(vocab.clone());
//! }
//! ```

pub mod utility;

mod bpe_trainer;
mod training_types;

#[doc(inline)]
pub use bpe_trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
#[doc(inline)]
pub use training_types::{CountType, StringChunkType};
