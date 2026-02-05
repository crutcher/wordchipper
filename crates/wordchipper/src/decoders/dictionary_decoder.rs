//! # Dictionary ``{ T -> Vec<u8> }`` Token Decoder

use crate::alloc::vec::Vec;
use crate::decoders::decode_results::DecodeResult;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::vocab_types::TokenSpanMap;

/// A token dictionary [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct DictionaryDecoder<T: TokenType> {
    /// Token to bytes mapping.
    ///
    /// Does not include byte-tokens.
    pub token_to_word: TokenSpanMap<T>,
}

impl<T: TokenType> DictionaryDecoder<T> {
    /// Build a [`DictionaryDecoder`] from this [`UnifiedTokenVocab`].
    ///
    /// ## Arguments
    /// * `unified_vocab` - The unified token vocabulary to build the decoder from.
    ///
    /// ## Returns
    /// A new `DictionaryDecoder` instance.
    pub fn from_unified_vocab(unified_vocab: UnifiedTokenVocab<T>) -> Self {
        Self::init(unified_vocab.unified_dictionary())
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `token_to_word` - The token to word mapping.
    ///
    /// ## Returns
    /// A new `DictionaryDecoder` instance.
    pub fn init(token_to_word: TokenSpanMap<T>) -> Self {
        Self { token_to_word }
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, tokens)))]
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<Vec<u8>>> {
        let capacity = (tokens.len() as f64 * EXPECTED_BYTES_PER_TOKEN) as usize;
        let mut value = Vec::with_capacity(capacity);
        let mut consumed = 0;
        for t in tokens {
            if let Some(w) = self.token_to_word.get(t) {
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
    use crate::decoders::utility::test_utils::common_decoder_unit_test;
    use crate::segmentation::SegmentationConfig;
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_dictionary_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        );

        let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());

        common_decoder_unit_test(vocab, &decoder);
    }
}
