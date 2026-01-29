//! # Parallel Decoder

use crate::decoders::{TokenDecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::utility::strings::string_from_lossy_utf8;

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
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        self.inner.incremental_decode(ctx)
    }

    fn try_decode_batch_to_bytes<V: AsRef<[T]>>(
        &self,
        batch: &[V],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;
        let batch: Vec<&[T]> = batch.iter().map(|v| v.as_ref()).collect();

        batch
            .into_par_iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect()
    }

    fn try_decode_batch_to_strings<V: AsRef<[T]>>(
        &self,
        batch: &[V],
    ) -> anyhow::Result<Vec<String>> {
        use rayon::prelude::*;

        let batch: Vec<&[T]> = batch.iter().map(|v| v.as_ref()).collect();

        batch
            .into_par_iter()
            .map(|tokens| {
                let buf = self.try_decode_to_bytes(tokens)?;
                Ok(string_from_lossy_utf8(buf))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::sync::Arc;
    use crate::decoders::utility::pair_decoder::PairExpansionDecoder;
    use crate::decoders::utility::test_utils::common_decoder_unit_test;
    use crate::segmentation::SegmentationConfig;
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;

    #[test]
    fn test_rayon_decoder() {
        type T = u16;

        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        )
        .into();

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab);

        let decoder = ParallelRayonDecoder::new(decoder);

        common_decoder_unit_test(vocab, &decoder);
    }
}
