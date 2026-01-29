//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::sync::Arc;
use crate::alloc::vec::Vec;
use crate::encoders::token_encoder::TokenEncoder;
use crate::regex::{RegexSupplierHandle, RegexWrapperHandle, default_regex_supplier};
use crate::segmentation::text_segmentor::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;

/// A Span-lookup / ``(T, T) -> T`` merge scan [`TokenEncoder`].
///
/// Builds a working set on the append buffer.
#[derive(Clone)]
pub struct MergeScanVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    /// Text Segmentor.
    pub segmentor: Arc<TextSegmentor>,
}

impl<T: TokenType> MergeScanVocabEncoder<T> {
    /// Construct an encoder from data.
    ///
    /// ## Arguments
    /// * `data` - The unified token vocabulary to build the encoder from.
    ///
    /// ## Returns
    /// A new `MergeHeapVocabEncoder` instance.
    pub fn init(data: Arc<UnifiedTokenVocab<T>>) -> Self {
        Self::init_with_factory(data, default_regex_supplier)
    }

    /// Construct an encoder from data.
    ///
    /// ## Arguments
    /// * `data` - The unified token vocabulary to build the encoder from.
    /// * `re_factory` - A factory function to create regex suppliers.
    ///
    /// ## Returns
    /// A new `MergeHeapVocabEncoder` instance.
    pub fn init_with_factory<F>(
        data: Arc<UnifiedTokenVocab<T>>,
        re_factory: F,
    ) -> Self
    where
        F: Fn(RegexWrapperHandle) -> RegexSupplierHandle,
    {
        let segmentor = TextSegmentor::from_config(data.segmentation.clone(), re_factory).into();

        Self { data, segmentor }
    }

    /// Compiler Hint.
    fn lookup_normal_token(
        &self,
        span: &[u8],
    ) -> Option<T> {
        self.data.lookup_token(span)
    }

    /// Compiler Hint.
    fn lookup_pair(
        &self,
        pair: &(T, T),
    ) -> Option<&T> {
        self.data.lookup_pair(pair)
    }

    /// Compiler Hint.
    fn append_tokens(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        self.data.byte_vocab().append_tokens(span, tokens);
    }
}

impl<T: TokenType> TokenEncoder<T> for MergeScanVocabEncoder<T> {
    fn segmentor(&self) -> &Arc<TextSegmentor> {
        &self.segmentor
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.data.segmentation.special_vocab()
    }

    fn encode_append_span_normal(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        if let Some(token) = self.lookup_normal_token(span) {
            // 1. Faster;
            // 2. Correct-or: Some words may not exist in the pair mappings.
            tokens.push(token);
            return;
        }

        // Reuse the output buffer as our working memory.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        self.append_tokens(span, tokens);

        // Incrementally shrink the working memory (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the lowest ranked merge available.
            if let Some((token, idx)) = tokens[start..]
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| self.lookup_pair(&(w[0], w[1])).map(|&token| (token, idx)))
                .min()
            {
                // Adjust the window index.
                let idx = start + idx;

                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                tokens[idx] = token;
                tokens.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::{DictionaryDecoder, TokenDecoder};
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::ByteMapVocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;
    use alloc::vec;

    #[test]
    fn test_encoder() {
        type T = u16;

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let byte_vocab: Arc<ByteMapVocab<T>> = Arc::new(Default::default());
        let segmentation = SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN);
        let vocab = build_test_vocab(byte_vocab.clone(), segmentation);

        let mut seg = vocab.segmentation.clone();
        seg.add_str_word("<|HI|>", 3000);

        let vocab: Arc<UnifiedTokenVocab<T>> =
            UnifiedTokenVocab::init(seg, vocab.span_vocab, vocab.pair_vocab).into();

        let special_sample = "hello <|HI|> world";

        let encoder = MergeScanVocabEncoder::<T>::init(vocab.clone());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        let decoder = DictionaryDecoder::from_unified_vocab(vocab);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        // Special handling.
        let tokens = encoder.encode(special_sample);
        assert_eq!(
            decoder.try_decode_to_string(tokens).unwrap(),
            special_sample
        );

        for sample in samples {
            let tokens = encoder.encode(sample);
            assert_eq!(decoder.try_decode_to_string(tokens).unwrap(), sample);
        }
    }
}
