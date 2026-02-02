//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::vec::Vec;
use crate::encoders::token_encoder::TokenEncoder;
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
    pub data: UnifiedTokenVocab<T>,

    /// Text Segmentor.
    pub segmentor: TextSegmentor,
}

impl<T: TokenType> MergeHeapVocabEncoder<T> {
    /// Construct an encoder from data.
    ///
    /// ## Arguments
    /// * `data` - The unified token vocabulary to build the encoder from.
    ///
    /// ## Returns
    /// A new `MergeHeapVocabEncoder` instance.
    pub fn init(data: UnifiedTokenVocab<T>) -> Self {
        let segmentor = TextSegmentor::from_config(data.segmentation.clone());

        Self { data, segmentor }
    }

    #[inline(always)]
    fn encode_append_span_normal(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // We reuse the output buffer as our working memory.
        // - `start` is the first index of the working memory buffer.
        let start = tokens.len();

        // Define CURRENT as `tokens[start..end]`.
        // - CURRENT[i] := tokens[start + i]
        self.data.byte_vocab().append_tokens(span, tokens);
        let mut end = tokens.len();

        // Define PAIR_RANKS as `tokens[end..]`
        // - there are `(end - start) - 1` items in PAIR_RANKS.
        // - PAIR_RANKS[i] := tokens[end + i]
        // - PAIR_RANKS[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))

        let get_pair_rank = {
            |tok: &mut [T], i: usize| {
                let pair = &(tok[start + i], tok[start + i + 1]);
                self.data
                    .lookup_pair(pair)
                    .unwrap_or_else(|| T::max_value())
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
    fn segmentor(&self) -> &TextSegmentor {
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
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()> {
        self.segmentor()
            .split_spans(text)
            .into_iter()
            .for_each(|span_ref| match span_ref {
                SpanRef::Normal(range) => {
                    let span = &text[range].as_bytes();
                    if let Some(token) = self.data.lookup_token(span) {
                        // 2. Correct-or: Some words may not exist in the pair mappings.
                        tokens.push(token);
                    } else {
                        self.encode_append_span_normal(span, tokens)
                    }
                }
                SpanRef::Special(range) => {
                    let span = &text[range].as_bytes();
                    let special_token = self.special_vocab().lookup_token(span).unwrap();
                    tokens.push(special_token);
                }
            });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::test_utils::{common_encoder_test_vocab, common_encoder_tests};

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();
        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone().into());
        common_encoder_tests(vocab.into(), &encoder)
    }

    #[test]
    fn test_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_encoder_u32() {
        test_encoder::<u32>();
    }
}
