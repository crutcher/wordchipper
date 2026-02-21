//! # Parallel Encoder

use crate::{
    TokenType,
    WCResult,
    alloc::sync::Arc,
    encoders::TokenEncoder,
    spanning::TextSpanner,
    vocab::SpecialVocab,
};

/// Batch-Level Parallel Encoder Wrapper.
///
/// Enables ``rayon`` encoding of batches when available.
pub struct ParallelRayonEncoder<T: TokenType> {
    /// Inner encoder.
    pub inner: Arc<dyn TokenEncoder<T>>,

    _marker: std::marker::PhantomData<T>,
}

impl<T> ParallelRayonEncoder<T>
where
    T: TokenType,
{
    /// Create a new parallel encoder.
    ///
    /// ## Arguments
    /// * `inner` - The token encoder to wrap.
    ///
    /// ## Returns
    /// A new `ParallelRayonEncoder` instance.
    pub fn new(inner: Arc<dyn TokenEncoder<T>>) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> TokenEncoder<T> for ParallelRayonEncoder<T>
where
    T: TokenType,
{
    fn spanner(&self) -> &Arc<dyn TextSpanner> {
        self.inner.spanner()
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.inner.special_vocab()
    }

    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> WCResult<()> {
        self.inner.try_encode_append(text, tokens)
    }

    fn try_encode_batch(
        &self,
        batch: &[&str],
    ) -> WCResult<Vec<Vec<T>>> {
        use rayon::prelude::*;

        let results: Vec<WCResult<Vec<T>>> = batch
            .par_iter()
            .map(|text| self.inner.try_encode(text))
            .collect();

        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenType,
        encoders::{
            TokenEncoder,
            testing::{common_encoder_test_vocab, common_encoder_tests},
        },
    };

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab::<T>();
        let inner = vocab.to_encoder_builder().with_parallel(false).build();
        let encoder = ParallelRayonEncoder::new(inner);

        assert_eq!(encoder.special_vocab(), encoder.inner.special_vocab());

        common_encoder_tests(vocab, Arc::new(encoder))
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
