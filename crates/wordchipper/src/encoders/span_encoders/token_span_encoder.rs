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
    ) -> anyhow::Result<()> {
        let mut se = (self.se_builder)();
        self.spanner.for_each_split_span(text, &mut |span_ref| {
            se.encode_append_span_ref(&self.vocab, text, span_ref, tokens);
            true
        });

        Ok(())
    }
}
