//! # Pair Expansion ``{ T -> (T, T) }`` Token Decoder

use crate::alloc::vec;
use crate::alloc::vec::Vec;
use crate::decoders::decode_results::DecodeResult;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::TokenType;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::vocab_types::TokenPairMap;
use crate::vocab::{ByteMapVocab, PairMapVocab};

/// A stack-based pair map `{T -> (T, T) }` incremental stack [`TokenDecoder`].
///
/// ## Style Hints
///
/// When there is no local ambiguity, instance names should prefer `decoder`;
/// and expand to `pair_decoder` when there is ambiguity.
#[derive(Clone)]
pub struct PairExpansionDecoder<T: TokenType> {
    /// Byte/token mapping table.
    pub byte_vocab: ByteMapVocab<T>,

    /// Token to pair mapping.
    pub token_pairs: TokenPairMap<T>,
}

impl<T: TokenType> PairExpansionDecoder<T> {
    /// Build a [`PairExpansionDecoder`] from this [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `pair_vocab` - The pair vocabulary mapping to build the decoder from.
    ///
    /// ## Returns
    /// A new `PairExpansionDecoder` instance.
    pub fn from_pair_vocab(pair_vocab: &PairMapVocab<T>) -> Self {
        let token_pairs = pair_vocab
            .pairs()
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();
        Self::init(pair_vocab.byte_vocab.clone(), token_pairs)
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `token_pairs` - The token to pair mapping.
    ///
    /// ## Returns
    /// A new `PairExpansionDecoder` instance.
    pub fn init(
        byte_vocab: ByteMapVocab<T>,
        token_pairs: TokenPairMap<T>,
    ) -> Self {
        Self {
            byte_vocab,
            token_pairs,
        }
    }
}

impl<T: TokenType> TokenDecoder<T> for PairExpansionDecoder<T> {
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<Vec<u8>>> {
        let capacity = (tokens.len() as f64 * EXPECTED_BYTES_PER_TOKEN) as usize;
        let mut value = Vec::with_capacity(capacity);

        let mut stack = vec![];
        let mut consumed = 0;

        for t in tokens {
            stack.push(*t);

            while let Some(t) = stack.pop() {
                if let Some(b) = self.byte_vocab.get_byte(t) {
                    value.push(b);
                } else if let Some((a, b)) = self.token_pairs.get(&t) {
                    stack.push(*b);
                    stack.push(*a);
                } else {
                    stack.push(t);
                    break;
                }
            }

            if stack.is_empty() {
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
    use crate::spanning::TextSpanningConfig;
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_pair_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        );

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab);

        assert_eq!(&decoder.byte_vocab, vocab.byte_vocab());

        common_decoder_unit_test(vocab, &decoder);
    }
}
