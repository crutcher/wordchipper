//! # Pair Expansion ``{ T -> (T, T) }`` Token Decoder

use crate::alloc::sync::Arc;
use crate::decoders::decode_context::TokenDecodeContext;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{TokenToPairMap, TokenType};
use crate::vocab::{ByteMapVocab, PairMapVocab};

/// A Pair Expansion ``{ T -> (T, T) }``  [`TokenDecoder`].
#[derive(Clone)]
pub struct PairExpansionDecoder<T: TokenType> {
    /// Byte/token mapping table.
    byte_vocab: Arc<ByteMapVocab<T>>,

    /// Token to pair mapping.
    token_map: TokenToPairMap<T>,
}

impl<T: TokenType> PairExpansionDecoder<T> {
    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `token_map` - The token to pair mapping.
    ///
    /// ## Returns
    /// A new `PairExpansionDecoder` instance.
    pub fn new<B>(
        byte_vocab: B,
        token_map: TokenToPairMap<T>,
    ) -> Self
    where
        B: Into<Arc<ByteMapVocab<T>>>,
    {
        Self {
            byte_vocab: byte_vocab.into(),
            token_map,
        }
    }

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
        Self::new(pair_vocab.byte_vocab().clone(), token_map)
    }

    /// Get the byte table.
    ///
    /// ## Returns
    /// A reference to the internal `ByteMapVocab` arc.
    pub fn byte_vocab(&self) -> &Arc<ByteMapVocab<T>> {
        &self.byte_vocab
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
    use crate::alloc::sync::Arc;
    use crate::decoders::utility::test_utils::common_decoder_unit_test;
    use crate::segmentation::SegmentationConfig;
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_pair_decoder() {
        type T = u16;

        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        )
        .into();

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab);

        assert_eq!(decoder.byte_vocab(), vocab.byte_vocab());

        common_decoder_unit_test(vocab, &decoder);
    }
}
