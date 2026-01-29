//! # Dictionary ``{ T -> Vec<u8> }`` Token Decoder

use crate::decoders::decode_context::TokenDecodeContext;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{TokenToWordMap, TokenType};
use crate::vocab::UnifiedTokenVocab;

/// A token dictionary [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct DictionaryDecoder<T: TokenType> {
    /// Token to bytes mapping.
    ///
    /// Does not include byte-tokens.
    pub token_to_word: TokenToWordMap<T>,
}

impl<T: TokenType> DictionaryDecoder<T> {
    /// Build a [`DictionaryDecoder`] from this [`UnifiedTokenVocab`].
    ///
    /// ## Arguments
    /// * `unified_vocab` - The unified token vocabulary to build the decoder from.
    ///
    /// ## Returns
    /// A new `DictionaryDecoder` instance.
    pub fn from_unified_vocab<V>(unified_vocab: V) -> Self
    where
        V: AsRef<UnifiedTokenVocab<T>>,
    {
        Self::init(unified_vocab.as_ref().unified_dictionary())
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `token_to_word` - The token to word mapping.
    ///
    /// ## Returns
    /// A new `DictionaryDecoder` instance.
    pub fn init(token_to_word: TokenToWordMap<T>) -> Self {
        Self { token_to_word }
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, ctx)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(w) = self.token_to_word.get(&t) {
                ctx.buf.extend_from_slice(w.as_slice());
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
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_dictionary_decoder() {
        type T = u16;

        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        )
        .into();

        let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());

        common_decoder_unit_test(vocab, &decoder);
    }
}
