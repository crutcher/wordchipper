//! # Token Encoder Trait

use crate::alloc::vec::Vec;
use crate::spanning::TextSpanner;
use crate::types::TokenType;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::special_vocab::SpecialVocab;

/// The common trait for `String/&[u8] -> Vec<T>` encoders.
///
/// ## Style Hints
///
/// When there is no local ambiguity with other encoders,
/// instance names for implementing types should prefer `encoder`;
/// and use the preferred name for the implementing type
/// when there is a conflict.
pub trait TokenEncoder<T: TokenType>: Clone + Send + Sync {
    /// Return the attached text segmentor.
    fn spanner(&self) -> &TextSpanner;

    /// Return the attached special vocab.
    ///
    /// ## Returns
    /// A reference to the internal `SpecialVocab`.
    fn special_vocab(&self) -> &SpecialVocab<T>;

    /// Encode bytes into tokens.
    ///
    /// ## Arguments
    /// * `text` - The string slice to encode.
    /// * `tokens` - The target token buffer to append to.
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()>;

    /// Return the expected bytes per token ratio.
    ///
    /// This is used by [`TokenEncoder::predict_token_buffer_size`] to predict
    /// the size needed when pre-allocating token buffers.
    fn expected_bytes_per_token(&self) -> f32 {
        EXPECTED_BYTES_PER_TOKEN
    }

    /// Predict the capacity needed when pre-allocating token buffers.
    ///
    /// See: [`TokenEncoder::expected_bytes_per_token`].
    fn predict_token_buffer_size(
        &self,
        text: &str,
    ) -> usize {
        ((text.len() as f32 * 1.1) / self.expected_bytes_per_token()) as usize
    }

    /// Encode text into tokens, returning an error if the encoding fails.
    ///
    /// ## Arguments
    /// * `text` - The text to encode.
    ///
    /// ## Returns
    /// A `Result` containing the vector of tokens or an error.
    fn try_encode(
        &self,
        text: &str,
    ) -> anyhow::Result<Vec<T>> {
        let capacity = self.predict_token_buffer_size(text);
        let mut tokens = Vec::with_capacity(capacity);

        self.try_encode_append(text, &mut tokens)?;
        Ok(tokens)
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
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>> {
        batch.iter().map(|s| self.try_encode(s)).collect()
    }
}
