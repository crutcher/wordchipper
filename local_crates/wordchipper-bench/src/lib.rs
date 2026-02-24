#![allow(missing_docs)]

use std::sync::{Arc, LazyLock, Mutex};

use wordchipper::{
    TokenEncoder,
    TokenEncoderOptions,
    TokenType,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    pretrained::openai::OATokenizer,
};

/// The huggingface/tokenizers model to use for `cl100k_base`.
pub const HF_CL100K: &str = "Xenova/text-embedding-ada-002";

/// The huggingface/tokenizers model to use for `o200k_base`.
pub const HF_O200K: &str = "Xenova/gpt-4o";

/// The shared disk cache for benchmarks.
#[allow(clippy::declare_interior_mutable_const)]
pub const DISK_CACHE: LazyLock<Arc<Mutex<WordchipperDiskCache>>> =
    LazyLock::new(|| Arc::new(Mutex::new(WordchipperDiskCache::default())));

/// Builds an `Arc<Tokenizer<T>>` for the target model and options.
///
/// Use the default disk cache to load the vocab.
pub fn load_vocab<T: TokenType>(model: OATokenizer) -> Arc<UnifiedTokenVocab<T>> {
    let binding = DISK_CACHE;
    let mut guard = binding.lock().unwrap();
    let disk_cache = &mut *guard;

    let vocab: Arc<UnifiedTokenVocab<T>> = model.load_vocab::<T>(disk_cache).unwrap().into();

    vocab
}
/// Builds an `Arc<Tokenizer<T>>` for the target model and options.
///
/// Use the default disk cache to load the vocab.
pub fn load_encoder<T: TokenType>(
    model: OATokenizer,
    options: TokenEncoderOptions,
) -> Arc<dyn TokenEncoder<T>> {
    options.build(load_vocab::<T>(model))
}
