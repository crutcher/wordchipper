//! `TokenDecoder` Builder

use crate::{
    TokenType,
    alloc::sync::Arc,
    decoders::{SlabIndexDecoder, TokenDecoder},
    vocab::UnifiedTokenVocab,
};

/// Builder for production [`TokenDecoder`]s.
#[derive(Clone, PartialEq)]
pub struct TokenDecoderBuilder<T: TokenType> {
    vocab: Arc<UnifiedTokenVocab<T>>,

    parallel: bool,
}

impl<T: TokenType> TokenDecoderBuilder<T> {
    /// Build a [`TokenDecoder`] with the default configuration.
    pub fn default(vocab: Arc<UnifiedTokenVocab<T>>) -> Arc<dyn TokenDecoder<T>> {
        Self::new(vocab).build()
    }

    /// Create a new `TokenDecoderBuilder`.
    pub fn new(vocab: Arc<UnifiedTokenVocab<T>>) -> Self {
        Self {
            vocab,
            parallel: true,
        }
    }

    /// Get whether the decoder should use parallel decoding.
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Set whether the decoder should use parallel decoding.
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.parallel = parallel;
        self
    }

    /// Build a `TokenDecoder` from the builder's state.
    pub fn build(&self) -> Arc<dyn TokenDecoder<T>> {
        #[allow(unused_mut)]
        let mut dec: Arc<dyn TokenDecoder<T>> =
            Arc::new(SlabIndexDecoder::from_vocab(self.vocab.clone()));

        #[cfg(feature = "rayon")]
        if self.parallel {
            use crate::support::concurrency::rayon::ParallelRayonDecoder;
            dec = Arc::new(ParallelRayonDecoder::new(dec));
        }

        dec
    }
}
