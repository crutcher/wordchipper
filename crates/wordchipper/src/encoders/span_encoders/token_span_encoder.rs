use crate::{
    TokenEncoder,
    TokenType,
    UnifiedTokenVocab,
    alloc::{boxed::Box, sync::Arc, vec::Vec},
    encoders::span_encoders::SpanEncoder,
    spanning::TextSpanner,
    vocab::SpecialVocab,
};

/// A [`TokenEncoder`] that composes a [`TextSpanner`] with a [`SpanEncoder`].
pub struct TokenSpanEncoder<T>
where
    T: TokenType,
{
    /// The reference vocabulary.
    vocab: UnifiedTokenVocab<T>,

    /// Text Spanner.
    spanner: Arc<dyn TextSpanner>,

    se_builder: Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync>,
}

impl<T: TokenType> TokenSpanEncoder<T> {
    /// Create a new encoder.
    pub fn new(
        spanner: Arc<dyn TextSpanner>,
        vocab: UnifiedTokenVocab<T>,
        se_builder: Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync>,
    ) -> Self {
        Self {
            vocab,
            spanner,
            se_builder,
        }
    }
}

impl<T: TokenType> TokenEncoder<T> for TokenSpanEncoder<T> {
    fn spanner(&self) -> Arc<dyn TextSpanner> {
        self.spanner.clone()
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.vocab.spanning().specials()
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, text, tokens))
    )]
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> crate::errors::WCResult<()> {
        let mut se = (self.se_builder)();

        self.spanner.for_each_split_span(text, &mut |span_ref| {
            // Note: .split_spans().into_iter().for_each() is *very* slightly faster
            // But, extending this interface to allow early exit via accepted specials
            // would end up slowing us down when that was enabled.

            se.encode_append_span_ref(&self.vocab, text, span_ref, tokens);
            true
        });

        Ok(())
    }
}
