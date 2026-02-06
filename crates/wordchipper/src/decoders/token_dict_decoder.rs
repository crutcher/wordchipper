//! # Dictionary ``{ T -> Vec<u8> }`` Token Decoder

use crate::alloc::vec::Vec;
use crate::decoders::decode_results::DecodeResult;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::vocab_types::TokenSpanMap;

/// A [`TokenDecoder<T>`] over a unified `{ T -> Vec<u8> }` dictionary.
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
        Self { token_spans }
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
        let capacity = (tokens.len() as f64 * EXPECTED_BYTES_PER_TOKEN) as usize;
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
    use crate::decoders::utility::testing::common_decoder_unit_test;
    use crate::pretrained::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::spanning::TextSpanningConfig;
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_dictionary_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        );

        let decoder = TokenDictDecoder::from_unified_vocab(vocab.clone());

        common_decoder_unit_test(vocab, &decoder);
    }
}
