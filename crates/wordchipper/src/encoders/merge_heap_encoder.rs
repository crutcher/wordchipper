//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::vec::Vec;
use crate::encoders::token_encoder::TokenEncoder;
use crate::segmentation::SpanRef;
use crate::segmentation::text_segmentor::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use core::num::NonZeroUsize;

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
    /// Intialize an encoder.
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
    pub fn encode_append_word(
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

        let pr_for_tokens = {
            |tok: &[T], a: usize, b: usize| {
                let pair = &(tok[start + a], tok[start + b]);
                self.data
                    .lookup_pair(pair)
                    .unwrap_or_else(|| T::max_value())
            }
        };

        // We keep the following property:
        // - pair_ranks[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))
        // - pair_ranks.len() = CURRENT.len() - 1 = end - start - 1
        pair_ranks.clear();
        pair_ranks.extend((0..(tokens.len() - start - 1)).map(|i| pr_for_tokens(tokens, i, i + 1)));

        while let Some((new_token, i)) = pair_ranks
            .iter()
            .enumerate()
            .filter_map(|(i, &new_token)| {
                if new_token == T::max_value() {
                    None
                } else {
                    Some((new_token, i))
                }
            })
            .min()
        {
            // At this point, i selects CURRENT[i], PAIR_RANKS[i] such that:
            // - PAIR_RANKS[i] != max_value
            // - PAIR_RANKS[i] is smallest

            // Set CURRENT[i] to the new target rank.
            tokens[start + i] = new_token;

            if i > 0 {
                // If there is a preceding token, recompute PAIR_RANKS[i-1].
                pair_ranks[i - 1] = pr_for_tokens(tokens, i - 1, i);
            }

            if i + 2 < tokens.len() - start {
                // If this pair rank exists,
                // it will become PAIR_RANKS[i] following the remove below.
                pair_ranks[i + 1] = pr_for_tokens(tokens, i, i + 2);
            }

            // Drop PAIR_RANKS[i] and CURRENT[i+1].
            pair_ranks.remove(i);
            tokens.remove(start + i + 1);
        }
    }

    /// Encodes a single [`SpanRef`]".
    ///
    /// ## Arguments
    /// * `text` - The source slice.
    /// * `span_ref` - The labeling and sub-slicing of a span in `text`.
    /// * `tokens` - The target token buffer to append to.
    /// * `pair_ranks` - Working space for pair ranks.
    pub fn encode_append_span_ref(
        &self,
        text: &str,
        span_ref: SpanRef,
        tokens: &mut Vec<T>,
        pair_vec: &mut Vec<T>,
    ) {
        match span_ref {
            SpanRef::Gap(_) => (),
            SpanRef::Word(range) => {
                let span = &text[range].as_bytes();
                if let Some(token) = self.data.lookup_token(span) {
                    // 1. Faster;
                    // 2. Correct-or: Some words may not exist in the pair mappings.
                    tokens.push(token);
                } else {
                    if pair_vec.len() < span.len() - 1 {
                        pair_vec.resize(span.len() - 1, T::max_value());
                    }
                    self.encode_append_word(span, tokens, pair_vec);
                }
            }
            SpanRef::Special(range) => {
                let span = &text[range].as_bytes();
                let special_token = self.special_vocab().lookup_token(span).unwrap();
                tokens.push(special_token);
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
        let mut pair_vec = Vec::with_capacity(16);
        self.segmentor().for_each_split(text, &mut |span_ref| {
            self.encode_append_span_ref(text, span_ref, tokens, &mut pair_vec);
            true
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
