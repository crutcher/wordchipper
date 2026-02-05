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
/// instance names for implementing types should prefer `decoder`;
/// and use the preferred name for the implementing type
/// when there is conflict.
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
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.5);
        let mut tokens = Vec::with_capacity(capacity as usize);

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
