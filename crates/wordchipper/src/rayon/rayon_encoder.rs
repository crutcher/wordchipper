//! # Parallel Encoder

use crate::encoders::TokenEncoder;
use crate::segmentation::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use std::sync::Arc;

/// Batch-Level Parallel Encoder Wrapper.
///
/// Enables ``rayon`` encoding of batches when available.
#[derive(Clone)]
pub struct ParallelRayonEncoder<T: TokenType, D: TokenEncoder<T>> {
    /// Inner encoder.
    pub inner: D,

    _marker: std::marker::PhantomData<T>,
}

impl<T, D> ParallelRayonEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    /// Create a new parallel encoder.
    ///
    /// ## Arguments
    /// * `inner` - The token encoder to wrap.
    ///
    /// ## Returns
    /// A new `ParallelRayonEncoder` instance.
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> TokenEncoder<T> for ParallelRayonEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    fn segmentor(&self) -> &Arc<TextSegmentor> {
        self.inner.segmentor()
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.inner.special_vocab()
    }

    fn encode_append_span_normal(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        self.inner.encode_append_span_normal(span, tokens)
    }

    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        use rayon::prelude::*;
        batch.par_iter().map(|text| self.encode(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::{DictionaryDecoder, TokenDecoder};
    use crate::encoders::{DefaultTokenEncoder, TokenEncoder};
    use crate::rayon::rayon_encoder::ParallelRayonEncoder;
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_vocab::ByteMapVocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;
    use std::sync::Arc;

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
        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(byte_vocab.clone(), segmentation)
            .with_special_words(vec![("<|HI|>", 3000)])
            .into();

        let special_sample = "hello <|HI|> world";

        let encoder = DefaultTokenEncoder::<T>::init(vocab.clone());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        let encoder = ParallelRayonEncoder::new(encoder);
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
