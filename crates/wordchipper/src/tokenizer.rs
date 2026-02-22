//! # Combined Tokenizer

use crate::{
    TokenDecoder,
    TokenDecoderBuilder,
    TokenEncoder,
    TokenEncoderBuilder,
    TokenType,
    UnifiedTokenVocab,
    WCResult,
    alloc::{string::String, sync::Arc, vec::Vec},
    decoders::{BatchDecodeResult, DecodeResult},
    spanning::TextSpanner,
};

/// Builder for [`Tokenizer`]s.
pub struct TokenizerBuilder<T: TokenType> {
    vocab: Arc<UnifiedTokenVocab<T>>,
    encoder: TokenEncoderBuilder<T>,
    decoder: TokenDecoderBuilder<T>,
}

impl<T: TokenType> TokenizerBuilder<T> {
    /// Create a new builder with the default configuration.
    pub fn default(vocab: Arc<UnifiedTokenVocab<T>>) -> Arc<Tokenizer<T>> {
        Self::new(vocab).build()
    }

    /// Create a new builder.
    pub fn new(vocab: Arc<UnifiedTokenVocab<T>>) -> Self {
        Self {
            vocab: vocab.clone(),
            encoder: TokenEncoderBuilder::new(vocab.clone()),
            decoder: TokenDecoderBuilder::new(vocab),
        }
    }

    /// Get the underlying vocabulary.
    pub fn vocab(&self) -> &Arc<UnifiedTokenVocab<T>> {
        &self.vocab
    }

    /// Get the encoder builder.
    pub fn encoder(&self) -> &TokenEncoderBuilder<T> {
        &self.encoder
    }

    /// Get the decoder builder.
    pub fn decoder(&self) -> &TokenDecoderBuilder<T> {
        &self.decoder
    }

    /// Get the encoder builder for mutable access.
    pub fn encoder_mut(&mut self) -> &mut TokenEncoderBuilder<T> {
        &mut self.encoder
    }

    /// Get the decoder builder for mutable access.
    pub fn decoder_mut(&mut self) -> &mut TokenDecoderBuilder<T> {
        &mut self.decoder
    }

    /// Build the tokenizer.
    pub fn build(&self) -> Arc<Tokenizer<T>> {
        Tokenizer::new(
            self.vocab.clone(),
            self.encoder.build(),
            self.decoder.build(),
        )
        .into()
    }
}

/// Unified Tokenizer.
///
/// Combines:
///  * [`UnifiedTokenVocab`],
///  * [`TokenEncoder`], and
///  * [`TokenDecoder`] wrappers.
#[derive(Clone)]
pub struct Tokenizer<T: TokenType> {
    vocab: Arc<UnifiedTokenVocab<T>>,
    encoder: Arc<dyn TokenEncoder<T>>,
    decoder: Arc<dyn TokenDecoder<T>>,
}

impl<T: TokenType> Tokenizer<T> {
    /// Create a new tokenizer.
    pub fn new(
        vocab: Arc<UnifiedTokenVocab<T>>,
        encoder: Arc<dyn TokenEncoder<T>>,
        decoder: Arc<dyn TokenDecoder<T>>,
    ) -> Self {
        Self {
            vocab,
            encoder,
            decoder,
        }
    }

    /// Get the underlying vocabulary.
    pub fn vocab(&self) -> &Arc<UnifiedTokenVocab<T>> {
        &self.vocab
    }

    /// Get the underlying encoder.
    pub fn encoder(&self) -> &Arc<dyn TokenEncoder<T>> {
        &self.encoder
    }

    /// Get the underlying decoder.
    pub fn decoder(&self) -> &Arc<dyn TokenDecoder<T>> {
        &self.decoder
    }
}

impl<T: TokenType> TokenEncoder<T> for Tokenizer<T> {
    fn spanner(&self) -> &Arc<dyn TextSpanner> {
        self.encoder.spanner()
    }

    fn special_vocab(&self) -> &crate::vocab::SpecialVocab<T> {
        self.encoder.special_vocab()
    }

    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> WCResult<()> {
        self.encoder.try_encode_append(text, tokens)
    }

    fn try_encode(
        &self,
        text: &str,
    ) -> WCResult<Vec<T>> {
        self.encoder.try_encode(text)
    }

    fn try_encode_batch(
        &self,
        batch: &[&str],
    ) -> WCResult<Vec<Vec<T>>> {
        self.encoder.try_encode_batch(batch)
    }
}

impl<T: TokenType> TokenDecoder<T> for Tokenizer<T> {
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<Vec<u8>>> {
        self.decoder.try_decode_to_bytes(tokens)
    }

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[&[T]],
    ) -> WCResult<BatchDecodeResult<Vec<u8>>> {
        self.decoder.try_decode_batch_to_bytes(batch)
    }

    fn try_decode_to_string(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<String>> {
        self.decoder.try_decode_to_string(tokens)
    }

    fn try_decode_batch_to_strings(
        &self,
        batch: &[&[T]],
    ) -> WCResult<BatchDecodeResult<String>> {
        self.decoder.try_decode_batch_to_strings(batch)
    }
}
