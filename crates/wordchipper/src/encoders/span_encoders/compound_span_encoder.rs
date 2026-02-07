//! # Abstract Base [`TokenEncoder`].

use crate::alloc::vec::Vec;
use crate::encoders::span_encoders::merge_heap_encoder::MergeHeapSpanPolicy;
use crate::encoders::span_encoders::span_policy::SpanPolicy;
use crate::encoders::token_encoder::TokenEncoder;
use crate::spanning::{SpanRef, TextSpanner};
use crate::types::TokenType;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use core::num::NonZeroUsize;

/// A [`TokenEncoder`] with pluggable [`SpanPolicy`]s.
///
/// This [`TokenEncoder`].
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
            expected_bytes_per_token: EXPECTED_BYTES_PER_TOKEN,
            marker: Default::default(),
        }
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
    /// * `span_ref` - The labeling and sub-slicing of a span in `text`.
    /// * `tokens` - The target token buffer to append to.
    /// * `span_policy` - The [`SpanPolicy`] context.
    fn encode_append_span_ref(
        &self,
        text: &str,
        span_ref: SpanRef,
        tokens: &mut Vec<T>,
        span_policy: &mut S,
    ) {
        match span_ref {
            SpanRef::Gap(_) => (),
            SpanRef::Word(range) => {
                let span = &text[range].as_bytes();
                if let Some(token) = self.vocab.lookup_token(span) {
                    // 1. Faster;
                    // 2. Correct-or: Some words may not exist in the pair mappings.
                    tokens.push(token);
                } else {
                    span_policy.encode_compound_span(&self.vocab, span, tokens);
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

impl<T: TokenType, S: SpanPolicy<T>> TokenEncoder<T> for CompoundSpanVocabEncoder<T, S> {
    fn spanner(&self) -> &TextSpanner {
        &self.spanner
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.vocab.spanning().special_vocab()
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
            self.encode_append_span_ref(text, span_ref, tokens, &mut span_policy);
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
        let encoder = CompoundSpanVocabEncoder::<T>::new(vocab.clone().into(), None);

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
