//! # Heap Merge Word Encoder
//!
//! Maintains a heap of the best available merges from the pair vocab,
//! iterates until no more merges remain.

use crate::alloc::vec::Vec;
use crate::encoders::span_encoder::{SpanEncoder, SpanEncoderVocabEncoder};
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;

/// A [`crate::encoders::TokenEncoder`] using [`MergeHeapSpanEncoder`].
///
/// This encoder builds and maintains a best-merge heap of potential merges,
/// to avoid secondary lookups in the pair vocab.
pub type MergeHeapVocabEncoder<T> = SpanEncoderVocabEncoder<T, MergeHeapSpanEncoder<T>>;

/// A [`SpanEncoder`] using a merge heap algorithm.
///
/// This encoder builds and maintains a best-merge heap of potential merges,
/// to avoid secondary lookups in the pair vocab.
#[derive(Default)]
pub struct MergeHeapSpanEncoder<T: TokenType> {
    pair_ranks: Vec<T>,
}

impl<T: TokenType> SpanEncoder<T> for MergeHeapSpanEncoder<T> {
    fn encode_append_span(
        &mut self,
        data: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // We reuse the output buffer as our working memory.
        // - `start` is the first index of the working memory buffer.
        let start = tokens.len();

        // Define CURRENT as `tokens[start..]`.
        // - CURRENT[i] := tokens[start + i]
        data.byte_vocab().append_tokens(span, tokens);

        let pr_for_tokens = {
            |tok: &[T], a: usize, b: usize| {
                data.lookup_pair(&(tok[start + a], tok[start + b]))
                    .unwrap_or(T::max_value())
            }
        };

        // We keep the following property:
        // - pair_ranks[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))
        // - pair_ranks.len() = CURRENT.len() - 1 = end - start - 1
        self.pair_ranks.clear();
        self.pair_ranks
            .extend((0..(tokens.len() - start - 1)).map(|i| pr_for_tokens(tokens, i, i + 1)));

        while let Some((new_token, i)) = self
            .pair_ranks
            .iter()
            .enumerate()
            .filter_map(|(i, &new_token)| {
                if new_token != T::max_value() {
                    Some((new_token, i))
                } else {
                    None
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
                self.pair_ranks[i - 1] = pr_for_tokens(tokens, i - 1, i);
            }

            if i + 2 < tokens.len() - start {
                // If this pair rank exists,
                // it will become PAIR_RANKS[i] following the remove below.
                self.pair_ranks[i + 1] = pr_for_tokens(tokens, i, i + 2);
            }

            // Drop PAIR_RANKS[i] and CURRENT[i+1].
            self.pair_ranks.remove(i);
            tokens.remove(start + i + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::test_utils::{common_encoder_test_vocab, common_encoder_tests};
    use crate::types::TokenType;

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
