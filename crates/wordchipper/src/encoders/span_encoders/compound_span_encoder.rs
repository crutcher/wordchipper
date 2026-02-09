//! # Abstract Base [`TokenEncoder`].

use core::num::NonZeroUsize;

use crate::{
    alloc::vec::Vec,
    encoders::{
        TokenEncoder,
        span_encoders::{MergeHeapSpanPolicy, SpanPolicy},
    },
    spanning::{SpanRef, TextSpanner},
    types::TokenType,
    vocab::{DEFAULT_BYTE_PER_TOKEN_RATIO, SpecialVocab, UnifiedTokenVocab},
};

/// A [`TokenEncoder`] with pluggable [`SpanPolicy`]s.
///
/// This encoder pre-allocates output buffers based upon the capacity
/// predicted by [`CompoundSpanVocabEncoder::predict_token_buffer_size`].
/// The behavior can be tuned by adjusting the `expected_bytes_per_token` parameter.
///
/// This [`TokenEncoder`] leverages [`TextSpanner`] to split text:
/// * [`SpanRef::Gap`] spans are ignored.
/// * [`SpanRef::Special`] spans are encoded using the [`SpecialVocab`].
/// * [`SpanRef::Word`] spans are:
///   - first checked for exact matches in the [`UnifiedTokenVocab`];
///   - falls back to the [`SpanPolicy`] context to encode the span.
///
/// The [`SpanPolicy`] provides a pluggable mechanism to swap different
/// encoder polices over compound word spans (those made up of more than one token).
///
/// ## Style Hints
///
/// When there is no local ambiguity with other encoders,
/// instance names for implementing types should prefer `decoder`;
/// and use the preferred name for the specialized [`SpanPolicy`] type,
/// `${policy name}_encoder` when there is a conflict.
pub struct CompoundSpanVocabEncoder<T, S = MergeHeapSpanPolicy<T>>
where
    T: TokenType,
    S: SpanPolicy<T>,
{
    /// The reference vocabulary.
    pub vocab: UnifiedTokenVocab<T>,

    /// Text Spanner.
    pub spanner: TextSpanner,

    expected_bytes_per_token: f32,

    marker: core::marker::PhantomData<fn() -> S>,
}

impl<T, S> Clone for CompoundSpanVocabEncoder<T, S>
where
    T: TokenType,
    S: SpanPolicy<T>,
{
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            spanner: self.spanner.clone(),
            expected_bytes_per_token: self.expected_bytes_per_token,
            marker: Default::default(),
        }
    }
}

impl<T: TokenType, S: SpanPolicy<T>> CompoundSpanVocabEncoder<T, S> {
    /// Initialize an encoder.
    ///
    /// ## Arguments
    /// * `vocab` - The unified token vocabulary to build the encoder from.
    ///
    /// ## Returns
    /// A new `MergeHeapVocabEncoder` instance.
    pub fn new(
        vocab: UnifiedTokenVocab<T>,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let spanner = TextSpanner::from_config(vocab.spanning().clone(), max_pool);

        Self {
            vocab,
            spanner,
            expected_bytes_per_token: DEFAULT_BYTE_PER_TOKEN_RATIO,
            marker: Default::default(),
        }
    }

    /// Get the expected bytes per token.
    pub fn expected_bytes_per_token(&self) -> f32 {
        self.expected_bytes_per_token
    }

    /// Set the expected bytes per token.
    ///
    /// This biases the size of pre-allocated encoding buffers.
    pub fn with_expected_bytes_per_token(
        mut self,
        expected: f32,
    ) -> Self {
        self.expected_bytes_per_token = expected;
        self
    }

    /// Encodes a single [`SpanRef`]".
    ///
    /// ## Arguments
    /// * `text` - The source slice.
    /// * `policy` - The [`SpanPolicy`] context.
    /// * `span_ref` - The labeling and sub-slicing of a span in `text`.
    /// * `tokens` - The target token buffer to append to.
    fn encode_append_span_ref(
        &self,
        text: &str,
        policy: &mut S,
        span_ref: SpanRef,
        tokens: &mut Vec<T>,
    ) {
        match span_ref {
            SpanRef::Word(range) => {
                let span = &text[range].as_bytes();
                if let Some(token) = self.vocab.lookup_token(span) {
                    // 1. Faster;
                    // 2. Correct-or: Some words may not exist in the pair mappings.
                    tokens.push(token);
                } else {
                    policy.encode_compound_span(&self.vocab, span, tokens);
                }
            }
            SpanRef::Special(range) => {
                let span = &text[range].as_bytes();
                let special_token = self.special_vocab().lookup_token(span).unwrap();
                tokens.push(special_token);
            }
            _ => (),
        }
    }
}

impl<T: TokenType, S: SpanPolicy<T>> TokenEncoder<T> for CompoundSpanVocabEncoder<T, S> {
    type Token = T;

    fn spanner(&self) -> &TextSpanner {
        &self.spanner
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.vocab.spanning().specials()
    }

    fn expected_bytes_per_token(&self) -> f32 {
        self.expected_bytes_per_token
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
        let mut span_policy: S = Default::default();
        self.spanner().for_each_split_span(text, &mut |span_ref| {
            self.encode_append_span_ref(text, &mut span_policy, span_ref, tokens);
            true
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::testing::{common_encoder_test_vocab, common_encoder_tests};

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();
        let encoder = CompoundSpanVocabEncoder::<T>::new(vocab.clone().into(), None)
            .with_expected_bytes_per_token(7.5);

        assert_eq!(encoder.expected_bytes_per_token(), 7.5);

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
