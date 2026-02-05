//! # Merge Scan Word Encoder
//!
//! Incrementally re-scans for the best available merge,
//! iterates until no more merges remain.

use crate::alloc::vec::Vec;
use crate::encoders::span_encoder::{CompoundSpanVocabEncoder, SpanPolicy};
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;

/// A [`CompoundSpanVocabEncoder`] using [`MergeScanCompoundPolicy`].
///
/// This encoder incrementally re-scans for the best available merge,
/// iterates until no more merges remain.
///
/// ## Style Hints
///
/// When there is no local ambiguity with other encoders,
/// instance names for implementing types should prefer `decoder`;
/// and prefer `merge_scan_encoder` when there is a conflict.
pub type MergeScanVocabEncoder<T> = CompoundSpanVocabEncoder<T, MergeScanCompoundPolicy<T>>;

/// A [`SpanPolicy`] which incrementally scans for merges.
///
/// This encoder incrementally re-scans for the best available merge,
/// iterates until no more merges remain.
///
/// ## Style Hints
///
/// When there is no local ambiguity with other encoders,
/// [`CompoundSpanVocabEncoder`] encoders specialized by
/// this policy should should prefer the instance name `decoder`;
/// and fall back to `merge_scan_encoder` when there is a conflict.
#[derive(Default)]
pub struct MergeScanCompoundPolicy<T: TokenType> {
    marker: core::marker::PhantomData<T>,
}

impl<T: TokenType> SpanPolicy<T> for MergeScanCompoundPolicy<T> {
    fn encode_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // Reuse the output buffer as our working memory.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        vocab.byte_vocab().append_tokens(span, tokens);

        // Incrementally shrink the working memory (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the lowest ranked merge available.
            if let Some((token, idx)) = tokens[start..]
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| vocab.lookup_pair(&(w[0], w[1])).map(|token| (token, idx)))
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
    use crate::encoders::test_utils::{common_encoder_test_vocab, common_encoder_tests};
    use crate::types::TokenType;

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();
        let encoder = MergeScanVocabEncoder::<T>::init(vocab.clone(), None);
        common_encoder_tests(vocab, &encoder)
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
