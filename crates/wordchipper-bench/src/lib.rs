#![allow(missing_docs)]

use std::sync::{Arc, LazyLock, Mutex};

use wordchipper::{
    TokenEncoderBuilder,
    TokenType,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    encoders::token_span_encoder::SpanEncoderSelector,
    pretrained::openai::OATokenizer,
};

/// The huggingface/tokenizers model to use for `cl100k_base`.
pub const HF_CL100K: &str = "Xenova/text-embedding-ada-002";

/// The huggingface/tokenizers model to use for `o200k_base`.
pub const HF_O200K: &str = "Xenova/gpt-4o";

pub const DISK_CACHE: LazyLock<Arc<Mutex<WordchipperDiskCache>>> =
    LazyLock::new(|| Arc::new(Mutex::new(WordchipperDiskCache::default())));

/// Returns a configured encoder builder for the given vocab and selector.
///
/// Uses the default disk cache to load the vocab.
pub fn encoder_builder<T: TokenType>(
    model: OATokenizer,
    selector: SpanEncoderSelector,
) -> TokenEncoderBuilder<T> {
    let cache = &*DISK_CACHE;
    let mut guard = cache.lock().unwrap();
    let disk_cache = &mut *guard;

    let vocab: Arc<UnifiedTokenVocab<T>> = model.load_vocab::<T>(disk_cache).unwrap().into();
    TokenEncoderBuilder::new(vocab).with_span_encoder(selector)
}
