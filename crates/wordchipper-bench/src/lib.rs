#![allow(missing_docs)]

use std::sync::Arc;

use wordchipper::{
    TokenEncoderBuilder,
    TokenType,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    encoders::token_span_encoder::SpanEncoderSelector,
    pretrained::openai::OATokenizer,
};

/// Returns a configured encoder builder for the given vocab and selector.
///
/// Uses the default disk cache to load the vocab.
pub fn encoder_builder<T: TokenType>(
    model: OATokenizer,
    selector: SpanEncoderSelector,
) -> TokenEncoderBuilder<T> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: Arc<UnifiedTokenVocab<T>> = model.load_vocab::<T>(&mut disk_cache).unwrap().into();

    TokenEncoderBuilder::new(vocab.clone()).with_span_encoder(selector)
}
