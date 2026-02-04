//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::vec::Vec;
use crate::encoders::merge_heap_encoder::MergeHeapSpanEncoder;
use crate::encoders::token_encoder::TokenEncoder;
use crate::segmentation::SpanRef;
use crate::segmentation::text_segmentor::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use core::num::NonZeroUsize;

/// Span encoder trait for [`SpanEncoderVocabEncoder`]
pub trait SpanEncoder<T: TokenType>: Default {
    /// Encodes a single normal "word".
    ///
    /// ## Arguments
    /// * `span` - The byte span to encode.
    /// * `tokens` - The target token buffer to append to.
    fn encode_append_span(
        &mut self,
        data: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    );
}

/// A [`TokenEncoder`] with pluggable [`SpanEncoder`]s.
pub struct SpanEncoderVocabEncoder<T, S = MergeHeapSpanEncoder<T>>
where
    T: TokenType,
    S: SpanEncoder<T>,
{
    /// Data for the encoders.
    pub data: UnifiedTokenVocab<T>,

    /// Text Segmentor.
    pub segmentor: TextSegmentor,

    marker: core::marker::PhantomData<fn() -> S>,
}

impl<T, S> Clone for SpanEncoderVocabEncoder<T, S>
where
    T: TokenType,
    S: SpanEncoder<T>,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            segmentor: self.segmentor.clone(),
            marker: Default::default(),
        }
    }
}

impl<T: TokenType, S: SpanEncoder<T>> SpanEncoderVocabEncoder<T, S> {
    /// Initialize an encoder.
    ///
    /// ## Arguments
    /// * `data` - The unified token vocabulary to build the encoder from.
    ///
    /// ## Returns
    /// A new `MergeHeapVocabEncoder` instance.
    pub fn init(
        data: UnifiedTokenVocab<T>,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let segmentor = TextSegmentor::from_config(data.segmentation.clone(), max_pool);

        Self {
            data,
            segmentor,
            marker: Default::default(),
        }
    }

    /// Encodes a single [`SpanRef`]".
    ///
    /// ## Arguments
    /// * `text` - The source slice.
    /// * `span_ref` - The labeling and sub-slicing of a span in `text`.
    /// * `tokens` - The target token buffer to append to.
    /// * `span_encoder` - The [`SpanEncoder`] context.
    fn encode_append_span_ref(
        &self,
        text: &str,
        span_ref: SpanRef,
        tokens: &mut Vec<T>,
        span_encoder: &mut S,
    ) {
        match span_ref {
            SpanRef::Gap(_) => (),
            SpanRef::Word(range) => {
                let span = &text[range].as_bytes();
                if let Some(token) = self.data.lookup_token(span) {
                    // 1. Faster;
                    // 2. Correct-or: Some words may not exist in the pair mappings.
                    tokens.push(token);
                } else {
                    span_encoder.encode_append_span(&self.data, span, tokens);
                }
            }
            SpanRef::Special(range) => {
                let span = &text[range].as_bytes();
                let special_token = self.special_vocab().lookup_token(span).unwrap();
                tokens.push(special_token);
            }
        }
    }
}

impl<T: TokenType, S: SpanEncoder<T>> TokenEncoder<T> for SpanEncoderVocabEncoder<T, S> {
    fn segmentor(&self) -> &TextSegmentor {
        &self.segmentor
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.data.segmentation.special_vocab()
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, text, tokens))
    )]
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()> {
        let mut span_encoder: S = Default::default();
        self.segmentor().for_each_split(text, &mut |span_ref| {
            self.encode_append_span_ref(text, span_ref, tokens, &mut span_encoder);
            true
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::test_utils::{common_encoder_test_vocab, common_encoder_tests};

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();
        let encoder = SpanEncoderVocabEncoder::<T>::init(vocab.clone().into(), None);
        common_encoder_tests(vocab.into(), &encoder)
    }

    #[test]
    fn test_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_encoder_u32() {
        test_encoder::<u32>();
    }
}
