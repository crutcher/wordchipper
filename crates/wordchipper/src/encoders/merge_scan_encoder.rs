//! # Encoder for [`UnifiedTokenVocab`].

use crate::alloc::vec::Vec;
use crate::encoders::token_encoder::TokenEncoder;
use crate::segmentation::SpanRef;
use crate::segmentation::text_segmentor::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use core::num::NonZeroUsize;

/// A Span-lookup / ``(T, T) -> T`` merge scan [`TokenEncoder`].
///
/// Builds a working set on the append buffer.
#[derive(Clone)]
pub struct MergeScanVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: UnifiedTokenVocab<T>,

    /// Text Segmentor.
    pub segmentor: TextSegmentor,
}

impl<T: TokenType> MergeScanVocabEncoder<T> {
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
    /// Iteratively re-scans for the best possible merges from the pair vocab,
    /// iterates until no more merges remain.
    ///
    /// - Assumes that the full span has already failed a span map lookup.
    /// - Appends tokens to `tokens`; uses `tokens` tail as working space.
    ///
    /// ## Arguments
    /// * `span` - The byte span to encode.
    /// * `tokens` - The target token buffer to append to.
    #[inline(always)]
    fn encode_append_word(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // Reuse the output buffer as our working memory.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        self.data.byte_vocab().append_tokens(span, tokens);

        // Incrementally shrink the working memory (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the lowest ranked merge available.
            if let Some((token, idx)) = tokens[start..]
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| {
                    self.data
                        .lookup_pair(&(w[0], w[1]))
                        .map(|token| (token, idx))
                })
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

impl<T: TokenType> TokenEncoder<T> for MergeScanVocabEncoder<T> {
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
                        self.encode_append_word(span, tokens)
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
