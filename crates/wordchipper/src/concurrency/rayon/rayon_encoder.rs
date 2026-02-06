//! # Parallel Encoder

use crate::encoders::TokenEncoder;
use crate::spanning::TextSpanner;
use crate::types::{CommonHashSet, TokenType};
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
    fn spanner(&self) -> &TextSpanner {
        self.inner.spanner()
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.inner.special_vocab()
    }

    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
        special_filter: Option<&CommonHashSet<T>>,
    ) -> anyhow::Result<usize> {
        self.inner.try_encode_append(text, tokens, special_filter)
    }

    fn try_encode_batch(
        &self,
        batch: &[&str],
        special_filter: Option<&CommonHashSet<T>>,
    ) -> anyhow::Result<Vec<Vec<T>>> {
        use rayon::prelude::*;

        let results: Vec<anyhow::Result<Vec<T>>> = batch
            .par_iter()
            .map(|text| self.inner.try_encode(text, special_filter))
            .collect();

        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::concurrency::rayon::rayon_encoder::ParallelRayonEncoder;
    use crate::encoders::testing::{common_encoder_test_vocab, common_encoder_tests};
    use crate::encoders::{DefaultTokenEncoder, TokenEncoder};
    use crate::types::TokenType;

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();

        let encoder = DefaultTokenEncoder::<T>::new(vocab.clone().into(), None);
        let encoder = ParallelRayonEncoder::new(encoder);

        assert_eq!(
            encoder.spanner().word_regex().as_str(),
            vocab.spanning().pattern.as_str()
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
