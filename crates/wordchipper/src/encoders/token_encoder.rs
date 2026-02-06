//! # Token Encoder Trait

use crate::alloc::vec::Vec;
use crate::spanning::TextSpanner;
use crate::types::{CommonHashSet, TokenType};
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
    /// * `special_filter` - The set of special tokens to accept, or `None` to accept all.
    /// * `tokens` - The target token buffer to append to.
    ///
    /// ## Returns
    /// The `Result<usize>` of bytes consumed.
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
        special_filter: Option<&CommonHashSet<T>>,
    ) -> anyhow::Result<usize>;

    /// Encode text into tokens, returning an error if the encoding fails.
    ///
    /// ## Arguments
    /// * `text` - The text to encode.
    /// * `special_filter` - The set of special tokens to accept, or `None` to accept all.
    ///
    /// ## Returns
    /// A `Result` containing the vector of tokens or an error.
    fn try_encode(
        &self,
        text: &str,
        special_filter: Option<&CommonHashSet<T>>,
    ) -> anyhow::Result<Vec<T>> {
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.5);
        let mut tokens = Vec::with_capacity(capacity as usize);

        self.try_encode_append(text, &mut tokens, special_filter)?;
        Ok(tokens)
    }

    /// Encode a batch of text into tokens, returning an error if the encoding fails.
    ///
    /// ## Arguments
    /// * `batch` - A slice of strings to encode.
    /// * `special_filter` - The set of special tokens to accept, or `None` to accept all.
    ///
    /// ## Returns
    /// A `Result` containing the vector of token vectors or an error.
    fn try_encode_batch(
        &self,
        batch: &[&str],
        special_filter: Option<&CommonHashSet<T>>,
    ) -> anyhow::Result<Vec<Vec<T>>> {
        batch
            .iter()
            .map(|s| self.try_encode(s, special_filter))
            .collect()
    }
}
