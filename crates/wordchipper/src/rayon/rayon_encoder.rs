//! # Parallel Encoder

use crate::encoders::TokenEncoder;
use crate::segmentation::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;

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
    fn segmentor(&self) -> &TextSegmentor {
        self.inner.segmentor()
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.inner.special_vocab()
    }

    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()> {
        self.inner.try_encode_append(text, tokens)
    }

    fn try_encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>> {
        use rayon::prelude::*;

        let results: Vec<anyhow::Result<Vec<T>>> = batch
            .par_iter()
            .map(|text| self.inner.try_encode(text))
            .collect();

        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::encoders::test_utils::{common_encoder_test_vocab, common_encoder_tests};
    use crate::encoders::{DefaultTokenEncoder, TokenEncoder};
    use crate::rayon::rayon_encoder::ParallelRayonEncoder;
    use crate::types::TokenType;

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();

        let encoder = DefaultTokenEncoder::<T>::init(vocab.clone().into(), None);
        let encoder = ParallelRayonEncoder::new(encoder);

        assert_eq!(
            encoder.segmentor().word_regex().as_str(),
            vocab.segmentation.pattern.as_str()
        );
        assert_eq!(encoder.special_vocab(), encoder.inner.special_vocab());

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
