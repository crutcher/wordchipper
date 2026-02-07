//! # Parallel Decoder

use crate::decoders::{BatchDecodeResult, DecodeResult, TokenDecoder};
use crate::types::TokenType;

/// Batch-Level Parallel Decoder Wrapper.
///
/// Enables ``rayon`` decoding of batches when available.
#[derive(Clone)]
pub struct ParallelRayonDecoder<T: TokenType, D: TokenDecoder<T>> {
    /// Wrapped decoder.
    pub inner: D,

    _marker: std::marker::PhantomData<T>,
}

impl<T, D> ParallelRayonDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    /// Create a new parallel token decoders.
    ///
    /// ## Arguments
    /// * `inner` - The token decoder to wrap.
    ///
    /// ## Returns
    /// A new `ParallelRayonDecoder` instance.
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> TokenDecoder<T> for ParallelRayonDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<Vec<u8>>> {
        self.inner.try_decode_to_bytes(tokens)
    }

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<BatchDecodeResult<Vec<u8>>> {
        use rayon::prelude::*;

        batch
            .par_iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect::<anyhow::Result<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }

    fn try_decode_batch_to_strings(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<BatchDecodeResult<String>> {
        use rayon::prelude::*;

        batch
            .par_iter()
            .map(|tokens| self.try_decode_to_string(tokens))
            .collect::<anyhow::Result<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::utility::PairExpansionDecoder;
    use crate::decoders::utility::testing::common_decoder_unit_test;
    use crate::pretrained::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::spanning::TextSpanningConfig;
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::utility::testing::build_test_shift_byte_vocab;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_rayon_decoder() {
        type T = u16;

        let vocab: UnifiedTokenVocab<T> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        );

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab());

        let decoder = ParallelRayonDecoder::new(decoder);

        common_decoder_unit_test(vocab, &decoder);
    }
}
