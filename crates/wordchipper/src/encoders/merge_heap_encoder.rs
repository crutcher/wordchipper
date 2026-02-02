//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::vec::Vec;
use crate::encoders::token_encoder::TokenEncoder;
use crate::segmentation::SpanRef;
use crate::segmentation::text_segmentor::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use std::num::NonZeroUsize;

/// A Span-lookup / ``(T, T) -> T`` merge heap [`TokenEncoder`].
///
/// Builds a working set on the append buffer.
///
/// More complex than [`super::merge_scan_encoder::MergeScanVocabEncoder`],
/// but triggers fewer pair lookups.
#[derive(Clone)]
pub struct MergeHeapVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    data: UnifiedTokenVocab<T>,

    /// Text Segmentor.
    segmentor: TextSegmentor,
}

impl<T: TokenType> MergeHeapVocabEncoder<T> {
    /// Construct an encoder from data.
    ///
    /// ## Arguments
    /// * `data` - The unified token vocabulary to build the encoder from.
    ///
    /// ## Returns
    /// A new `MergeHeapVocabEncoder` instance.
    pub fn init(
        data: UnifiedTokenVocab<T>,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let segmentor = TextSegmentor::from_config(data.segmentation.clone(), max_pool);

        Self { data, segmentor }
    }

    /// Encodes a single normal "word".
    ///
    /// Maintains a heap of the best possible merges from the pair vocab,
    /// iterates until no more merges remain.
    ///
    /// - Assumes that the full span has already failed a span map lookup.
    /// - Appends tokens to `tokens`; uses `tokens` tail as working space.
    /// - Uses `pair_ranks` as working space for pair ranks.
    ///
    /// ## Arguments
    /// * `span` - The byte span to encode.
    /// * `tokens` - The target token buffer to append to.
    /// * `pair_ranks` - Working space for pair ranks.
    #[inline(always)]
    fn encode_append_word(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
        pair_ranks: &mut Vec<T>,
    ) {
        // We reuse the output buffer as our working memory.
        // - `start` is the first index of the working memory buffer.
        let start = tokens.len();

        // Define CURRENT as `tokens[start..]`.
        // - CURRENT[i] := tokens[start + i]
        self.data.byte_vocab().append_tokens(span, tokens);

        let get_pair_rank = {
            |tok: &mut [T], i: usize| {
                let pair = &(tok[start + i], tok[start + i + 1]);
                self.data
                    .lookup_pair(pair)
                    .unwrap_or_else(|| T::max_value())
            }
        };

        // We keep the following property:
        // - pair_ranks[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))
        // - pair_ranks.len() = CURRENT.len() - 1 = end - start - 1
        pair_ranks.clear();
        pair_ranks.extend(
            (0..(tokens.len() - start - 1))
                .into_iter()
                .map(|i| get_pair_rank(tokens, i)),
        );

        while let Some((t, i)) = pair_ranks
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
                pair_ranks[i - 1] = get_pair_rank(tokens, i - 1);
            }

            // Drop PAIR_RANKS[i] and CURRENT[i+1].
            // Order matters here for the indices.
            pair_ranks.remove(i);
            tokens.remove(start + i + 1);

            if i + 1 < tokens.len() - start {
                // If there is a following token, recompute PAIR_RANKS[i].
                pair_ranks[i] = get_pair_rank(tokens, i);
            }
        }
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
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()> {
        let mut pairs = vec![];

        self.segmentor()
            .split_spans(text)
            .into_iter()
            .for_each(|span_ref| match span_ref {
                SpanRef::Gap(_) => (),
                SpanRef::Word(range) => {
                    let span = &text[range].as_bytes();
                    if let Some(token) = self.data.lookup_token(span) {
                        // 1. Faster;
                        // 2. Correct-or: Some words may not exist in the pair mappings.
                        tokens.push(token);
                    } else {
                        self.encode_append_word(span, tokens, &mut pairs);
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
        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone().into(), None);
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
