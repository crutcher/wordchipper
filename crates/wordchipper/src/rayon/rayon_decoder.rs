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

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;

        batch
            .into_par_iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect()
    }

    fn try_decode_batch_to_strings(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<String>> {
        use rayon::prelude::*;

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
    use crate::decoders::DictionaryDecoder;
    use crate::encoders::MergeHeapVocabEncoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_vocab::ByteMapVocab;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::utility::testing::build_test_vocab;
    use num_traits::FromPrimitive;

    #[test]
    fn test_decoder() {
        type T = u16;

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let byte_vocab: Arc<ByteMapVocab<T>> = Arc::new(Default::default());
        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            byte_vocab.clone(),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        )
        .into();

        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone());

        let decoder = ParallelRayonDecoder::new(DictionaryDecoder::from_unified_vocab(vocab));
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples.iter() {
            let tokens = encoder.encode(sample);
            let decoded = decoder.try_decode_to_string(&tokens).unwrap();
            assert_eq!(&decoded, sample);
        }

        let token_batch: Vec<Vec<T>> = samples
            .iter()
            .map(|s| {
                s.as_bytes()
                    .iter()
                    .map(|b| T::from_u8(*b).unwrap())
                    .collect()
            })
            .collect();

        // Test the batch interfaces.
        let string_batch = decoder.try_decode_batch_to_strings(&token_batch).unwrap();
        assert_eq!(string_batch, samples);
    }
}
