//! # Token Encoder Trait

use crate::alloc::string::String;
use crate::alloc::sync::Arc;
use crate::alloc::vec::Vec;
use crate::segmentation::TextSegmentor;
use crate::segmentation::text_segmentor::SpanRef;
use crate::types::TokenType;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::special_vocab::SpecialVocab;

/// A trait for token encoders.
pub trait TokenEncoder<T: TokenType>: Send + Sync {
    /// Return the attached text segmentor.
    ///
    /// ## Returns
    /// A reference to the internal `TextSegmentor` arc.
    fn segmentor(&self) -> &Arc<TextSegmentor>;

    /// Return the attached special vocab.
    ///
    /// ## Returns
    /// A reference to the internal `SpecialVocab`.
    fn special_vocab(&self) -> &SpecialVocab<T>;

    /// Encode a span appending to a target buffer.
    ///
    /// ## Arguments
    /// * `span` - The byte span to encode.
    /// * `tokens` - The target token buffer to append to.
    fn encode_append_span_normal(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    );

    /// Encode bytes into tokens.
    ///
    /// ## Arguments
    /// * `text` - The string slice to encode.
    /// * `tokens` - The target token buffer to append to.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) {
        self.segmentor()
            .split_spans(text)
            .into_iter()
            .for_each(|span_ref| match span_ref {
                SpanRef::Normal(span_str) => {
                    self.encode_append_span_normal(span_str.as_bytes(), tokens)
                }
                SpanRef::Special(s) => {
                    tokens.push(self.special_vocab().lookup_token(s.as_bytes()).unwrap());
                }
            });
    }

    /// Encode text into tokens.
    ///
    /// ## Arguments
    /// * `text` - The text to encode.
    ///
    /// ## Returns
    /// A vector of tokens.
    ///
    /// ## Panics
    /// Panics if encoding fails.
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        self.try_encode(text).unwrap()
    }

    /// Encode text into tokens, returning an error if the encoding fails.
    ///
    /// ## Arguments
    /// * `text` - The text to encode.
    ///
    /// ## Returns
    /// A `Result` containing the vector of tokens or an error.
    fn try_encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> anyhow::Result<Vec<T>> {
        let text = text.as_ref();
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.5);
        let mut tokens = Vec::with_capacity(capacity as usize);

        self.encode_append(text, &mut tokens);
        Ok(tokens)
    }

    /// Encode a batch of text into tokens.
    ///
    /// ## Arguments
    /// * `batch` - A slice of strings to encode.
    ///
    /// ## Returns
    /// A vector of token vectors.
    ///
    /// ## Panics
    /// Panics if any encoding fails.
    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        self.try_encode_batch(batch).unwrap()
    }

    /// Encode a batch of text into tokens, returning an error if the encoding fails.
    ///
    /// ## Arguments
    /// * `batch` - A slice of strings to encode.
    ///
    /// ## Returns
    /// A `Result` containing the vector of token vectors or an error.
    fn try_encode_batch(
        &self,
        batch: &[String],
    ) -> anyhow::Result<Vec<Vec<T>>> {
        batch.iter().map(|s| self.try_encode(s)).collect()
    }
}
