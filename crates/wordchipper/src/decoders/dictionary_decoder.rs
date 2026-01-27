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
    use crate::alloc::vec;
    use crate::encoders::merge_heap_encoder::MergeHeapVocabEncoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::byte_vocab::ByteMapVocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_dictionary_decoder() {
        type T = u16;

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let byte_vocab: Arc<ByteMapVocab<T>> = Arc::new(Default::default());
        let segmentation = SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN);

        let vocab: Arc<UnifiedTokenVocab<T>> =
            build_test_vocab(byte_vocab.clone(), segmentation).into();

        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone());

        let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.try_decode_to_string(&tokens).unwrap();
            assert_eq!(decoded, sample);
        }
    }
}
