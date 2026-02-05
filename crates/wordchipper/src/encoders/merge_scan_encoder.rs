//! # Merge Scan Word Encoder
//!
//! Incrementally re-scans for the best available merge,
//! iterates until no more merges remain.

use crate::alloc::vec::Vec;
use crate::encoders::span_encoder::{SpanEncoder, SpanEncoderVocabEncoder};
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;

/// A [`crate::encoders::TokenEncoder`] using [`MergeScanSpanEncoder`].
///
/// This encoder incrementally re-scans for the best available merge,
/// iterates until no more merges remain.
pub type MergeScanVocabEncoder<T> = SpanEncoderVocabEncoder<T, MergeScanSpanEncoder<T>>;

/// A [`SpanEncoder`] which incrementally scans for merges.
///
/// This encoder incrementally re-scans for the best available merge,
/// iterates until no more merges remain.
#[derive(Default)]
pub struct MergeScanSpanEncoder<T: TokenType> {
    marker: core::marker::PhantomData<T>,
}

impl<T: TokenType> SpanEncoder<T> for MergeScanSpanEncoder<T> {
    fn encode_append_span(
        &mut self,
        data: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // Reuse the output buffer as our working memory.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        data.byte_vocab().append_tokens(span, tokens);

        // Incrementally shrink the working memory (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the lowest ranked merge available.
            if let Some((token, idx)) = tokens[start..]
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| data.lookup_pair(&(w[0], w[1])).map(|token| (token, idx)))
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
