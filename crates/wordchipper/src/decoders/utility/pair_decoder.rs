//! # Pair Expansion ``{ T -> (T, T) }`` Token Decoder

use crate::decoders::decode_context::TokenDecodeContext;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::TokenType;
use crate::vocab::vocab_types::TokenPairMap;
use crate::vocab::{ByteMapVocab, PairMapVocab};

/// A [`TokenDecoder`] based on a token expansion map ``{ T -> (T, T) }``.
#[derive(Clone)]
pub struct PairExpansionDecoder<T: TokenType> {
    /// Byte/token mapping table.
    pub byte_vocab: ByteMapVocab<T>,

    /// Token to pair mapping.
    pub token_map: TokenPairMap<T>,
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
        let token_map = pair_vocab
            .pairs()
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();
        Self::init(pair_vocab.byte_vocab.clone(), token_map)
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `token_map` - The token to pair mapping.
    ///
    /// ## Returns
    /// A new `PairExpansionDecoder` instance.
    pub fn init(
        byte_vocab: ByteMapVocab<T>,
        token_map: TokenPairMap<T>,
    ) -> Self {
        Self {
            byte_vocab,
            token_map,
        }
    }
}

impl<T: TokenType> TokenDecoder<T> for PairExpansionDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, ctx)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = self.byte_vocab.get_byte(t) {
                ctx.buf.push(b);
            } else if let Some((a, b)) = self.token_map.get(&t) {
                ctx.stack.push(*b);
                ctx.stack.push(*a);
            } else {
                ctx.stack.push(t);
                break;
            }
        }
        ctx.stack.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::utility::test_utils::common_decoder_unit_test;
    use crate::segmentation::SegmentationConfig;
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_pair_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        );

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab);

        assert_eq!(&decoder.byte_vocab, vocab.byte_vocab());

        common_decoder_unit_test(vocab, &decoder);
    }
}
