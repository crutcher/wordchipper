//! # Dictionary ``{ T -> Vec<u8> }`` Token Decoder

use crate::{
    alloc::vec::Vec,
    decoders::{DecodeResult, TokenDecoder},
    types::TokenType,
    vocab::{DEFAULT_BYTE_PER_TOKEN_RATIO, TokenSpanMap, UnifiedTokenVocab},
};

/// A [`TokenDecoder<T>`] over a unified `{ T -> Vec<u8> }` dictionary.
///
/// It is expected that all tokens (single-byte and multibyte words,
/// and special tokens) are stored in the dictionary.
///
/// ## Style Hints
///
/// When there is no local ambiguity, instance names should prefer `decoder`;
/// and expand to `dict_decoder` when there is ambiguity.
#[derive(Clone)]
pub struct TokenDictDecoder<T: TokenType> {
    /// Token to bytes mapping.
    ///
    /// Does not include byte-tokens.
    token_spans: TokenSpanMap<T>,

    expected_bytes_per_token: f32,
}

impl<T: TokenType> TokenDictDecoder<T> {
    /// Build a [`TokenDictDecoder`] from this [`UnifiedTokenVocab`].
    ///
    /// ## Arguments
    /// * `unified_vocab` - The unified token vocabulary to build the decoder from.
    pub fn from_unified_vocab(unified_vocab: UnifiedTokenVocab<T>) -> Self {
        Self::new(unified_vocab.unified_dictionary())
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `token_spans` - The token to word mapping.
    pub fn new(token_spans: TokenSpanMap<T>) -> Self {
        Self {
            token_spans,
            expected_bytes_per_token: DEFAULT_BYTE_PER_TOKEN_RATIO,
        }
    }

    /// Get the expected bytes per token.
    pub fn expected_bytes_per_token(&self) -> f32 {
        self.expected_bytes_per_token
    }

    /// Sets the expected bytes per token.
    ///
    /// This is used to bias the capacity of the output buffer in `try_decode_to_bytes`.
    pub fn with_expected_bytes_per_token(
        mut self,
        expected: f32,
    ) -> Self {
        self.expected_bytes_per_token = expected;
        self
    }

    /// Predict the capacity needed when pre-allocating output buffers.
    pub fn predicted_byte_buffer_size(
        &self,
        tokens: &[T],
    ) -> usize {
        (tokens.len() as f32 * 1.1 * self.expected_bytes_per_token) as usize
    }

    /// Get the [`TokenSpanMap`].
    pub fn token_spans(&self) -> &TokenSpanMap<T> {
        &self.token_spans
    }
}

impl<T: TokenType> TokenDecoder<T> for TokenDictDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, tokens)))]
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<Vec<u8>>> {
        let capacity = self.predicted_byte_buffer_size(tokens);
        let mut value = Vec::with_capacity(capacity);

        let mut consumed = 0;
        for t in tokens {
            if let Some(w) = self.token_spans.get(t) {
                value.extend(w);
                consumed += 1;
            } else {
                break;
            }
        }
        Ok(DecodeResult::new(value, Some(tokens.len() - consumed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decoders::utility::testing::common_decoder_unit_test,
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanning::TextSpanningConfig,
        vocab::{
            UnifiedTokenVocab,
            utility::testing::{build_test_shift_byte_vocab, build_test_vocab},
        },
    };

    #[test]
    fn test_dictionary_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN),
        );

        let decoder =
            TokenDictDecoder::from_unified_vocab(vocab.clone()).with_expected_bytes_per_token(7.5);

        assert_eq!(decoder.expected_bytes_per_token(), 7.5);

        assert_eq!(decoder.token_spans(), &decoder.token_spans);

        common_decoder_unit_test(vocab, &decoder);
    }
}
