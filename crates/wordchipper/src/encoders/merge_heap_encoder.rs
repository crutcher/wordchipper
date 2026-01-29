//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::sync::Arc;
use crate::alloc::vec::Vec;
use crate::encoders::token_encoder::TokenEncoder;
use crate::regex::{RegexSupplierHandle, RegexWrapperHandle, default_regex_supplier};
use crate::segmentation::SpanRef;
use crate::segmentation::text_segmentor::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;

/// A Span-lookup / ``(T, T) -> T`` merge heap [`TokenEncoder`].
///
/// Builds a working set on the append buffer.
///
/// More complex than [`super::merge_scan_encoder::MergeScanVocabEncoder`],
/// but triggers fewer pair lookups.
#[derive(Clone)]
pub struct MergeHeapVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    /// Text Segmentor.
    pub segmentor: Arc<TextSegmentor>,
}

impl<T: TokenType> MergeHeapVocabEncoder<T> {
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

        // We reuse the output buffer as our working memory.
        // - `start` is the first index of the working memory buffer.
        let start = tokens.len();

        // Define CURRENT as `tokens[start..end]`.
        // - CURRENT[i] := tokens[start + i]
        self.append_tokens(span, tokens);
        let mut end = tokens.len();

        // Define PAIR_RANKS as `tokens[end..]`
        // - there are `(end - start) - 1` items in PAIR_RANKS.
        // - PAIR_RANKS[i] := tokens[end + i]
        // - PAIR_RANKS[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))

        let get_pair_rank = {
            |tok: &mut [T], i: usize| {
                let pair = &(tok[start + i], tok[start + i + 1]);
                match self.lookup_pair(pair) {
                    Some(&token) => token,
                    None => T::max_value(),
                }
            }
        };

        for i in 1..(end - start) {
            let rank = get_pair_rank(tokens, i - 1);
            tokens.push(rank);
        }

        while let Some((t, i)) = tokens[end..]
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| {
                if t == T::max_value() {
                    None
                } else {
                    Some((t, i))
                }
            })
            .min()
        {
            // At this point, i selects CURRENT[i], PAIR_RANKS[i] such that:
            // - PAIR_RANKS[i] != max_value
            // - PAIR_RANKS[i] is smallest

            // We need to merge CURRENT[i..=i+1] and PAIR_RANKS[i..=i+1]

            // Set CURRENT[i] to the new target rank.
            tokens[start + i] = t;

            if i > 0 {
                // If there is a preceding token, recompute PAIR_RANKS[i-1].
                tokens[end + i - 1] = get_pair_rank(tokens, i - 1);
            }

            // Drop PAIR_RANKS[i] and CURRENT[i+1].
            // Order matters here for the indices.
            tokens.remove(end + i);
            tokens.remove(start + i + 1);

            end -= 1;

            if end + i < tokens.len() {
                // If there is a following token, recompute PAIR_RANKS[i].
                tokens[end + i] = get_pair_rank(tokens, i);
            }
        }

        // Drop the PAIR_RANKS buffer.
        tokens.truncate(end);
    }
}

impl<T: TokenType> TokenEncoder<T> for MergeHeapVocabEncoder<T> {
    fn segmentor(&self) -> &Arc<TextSegmentor> {
        &self.segmentor
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.data.segmentation.special_vocab()
    }

    /// Encode bytes into tokens.
    ///
    /// ## Arguments
    /// * `text` - The string slice to encode.
    /// * `tokens` - The target token buffer to append to.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()> {
        self.segmentor()
            .split_spans(text)
            .into_iter()
            .for_each(|span_ref| match span_ref {
                SpanRef::Normal(span_str) => {
                    self.encode_append_span_normal(span_str.as_bytes(), tokens)
                }
                SpanRef::Special(s) => {
                    tokens.push(self.special_vocab().lookup_token(s.as_bytes()).unwrap());
                }
            });

        Ok(())
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
        seg.special_vocab_mut().add_str_word("<|HI|>", 3000);

        let vocab: Arc<UnifiedTokenVocab<T>> =
            UnifiedTokenVocab::init(seg, vocab.span_vocab, vocab.pair_vocab).into();

        let special_sample = "hello <|HI|> world";

        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        let decoder = DictionaryDecoder::from_unified_vocab(vocab);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        // Special handling.
        let tokens = encoder.try_encode(special_sample).unwrap();
        assert_eq!(
            decoder.try_decode_to_string(tokens).unwrap(),
            special_sample
        );

        for sample in samples {
            let tokens = encoder.try_encode(sample).unwrap();
            assert_eq!(decoder.try_decode_to_string(tokens).unwrap(), sample);
        }
    }
}
