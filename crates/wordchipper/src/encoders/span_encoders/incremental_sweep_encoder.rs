//! # Incremental merge-sweep [`SpanEncoder`].
//!
//! Incrementally re-scans for the best available merge,
//! iterates until no more merges remain.

use crate::{
    TokenType,
    alloc::vec::Vec,
    encoders::span_encoders::span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

/// A [`SpanEncoder`] which incrementally scans for merges.
///
/// This encoder incrementally re-scans for the best available merge,
/// iterates until no more merges remain.
#[derive(Default, Debug, Clone)]
pub struct IncrementalSweepSpanEncoder<T: TokenType> {
    marker: core::marker::PhantomData<T>,
}

impl<T: TokenType> SpanEncoder<T> for IncrementalSweepSpanEncoder<T> {
    fn encode_append_compound_span(
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
    use crate::{
        TokenType,
        alloc::{boxed::Box, sync::Arc},
        encoders::{
            span_encoders::TokenSpanEncoder,
            testing::{common_encoder_test_vocab, common_encoder_tests},
        },
    };

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();
        let encoder = TokenSpanEncoder::<T>::new(
            vocab.to_default_spanner(),
            vocab.clone(),
            Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<T>::default())),
        );
        common_encoder_tests(vocab, encoder)
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
