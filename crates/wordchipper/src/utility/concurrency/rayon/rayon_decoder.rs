//! # Parallel Decoder

use crate::{
    alloc::sync::Arc,
    decoders::{BatchDecodeResult, DecodeResult, TokenDecoder},
    types::TokenType,
};

/// Batch-Level Parallel Decoder Wrapper.
///
/// Enables ``rayon`` decoding of batches when available.
pub struct ParallelRayonDecoder<T: TokenType> {
    /// Wrapped decoder.
    pub inner: Arc<dyn TokenDecoder<T>>,

    _marker: std::marker::PhantomData<T>,
}

impl<T> ParallelRayonDecoder<T>
where
    T: TokenType,
{
    /// Create a new parallel token decoders.
    ///
    /// ## Arguments
    /// * `inner` - The token decoder to wrap.
    ///
    /// ## Returns
    /// A new `ParallelRayonDecoder` instance.
    pub fn new(inner: Arc<dyn TokenDecoder<T>>) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> TokenDecoder<T> for ParallelRayonDecoder<T>
where
    T: TokenType,
{
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> crate::errors::Result<DecodeResult<Vec<u8>>> {
        self.inner.try_decode_to_bytes(tokens)
    }

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[&[T]],
    ) -> crate::errors::Result<BatchDecodeResult<Vec<u8>>> {
        use rayon::prelude::*;

        batch
            .par_iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect::<crate::errors::Result<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }

    fn try_decode_batch_to_strings(
        &self,
        batch: &[&[T]],
    ) -> crate::errors::Result<BatchDecodeResult<String>> {
        use rayon::prelude::*;

        batch
            .par_iter()
            .map(|tokens| self.try_decode_to_string(tokens))
            .collect::<crate::errors::Result<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        UnifiedTokenVocab,
        decoders::utility::testing::common_decoder_unit_test,
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanners::TextSpanningConfig,
        vocab::utility::testing::{build_test_shift_byte_vocab, build_test_vocab},
    };

    #[test]
    fn test_rayon_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN),
        );

        let inner = vocab.to_decoder_builder().with_parallel(false).init();
        let decoder = ParallelRayonDecoder::new(inner);

        common_decoder_unit_test(vocab, &decoder);
    }
}
