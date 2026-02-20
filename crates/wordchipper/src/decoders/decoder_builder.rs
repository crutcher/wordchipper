//! `TokenDecoder` Builder

use crate::{
    alloc::sync::Arc,
    decoders::{SlabIndexDecoder, TokenDecoder},
    types::TokenType,
    vocab::UnifiedTokenVocab,
};

/// Builder for production [`TokenDecoder`]s.
#[derive(Clone, PartialEq)]
pub struct TokenDecoderBuilder<T: TokenType> {
    vocab: UnifiedTokenVocab<T>,

    parallel: bool,
}

impl<T: TokenType> TokenDecoderBuilder<T> {
    /// Build a [`TokenDecoder`] with the default configuration.
    pub fn default(vocab: UnifiedTokenVocab<T>) -> Arc<dyn TokenDecoder<T>> {
        Self::new(vocab).init()
    }

    /// Create a new `TokenDecoderBuilder`.
    pub fn new(vocab: UnifiedTokenVocab<T>) -> Self {
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
    pub fn init(self) -> Arc<dyn TokenDecoder<T>> {
        #[allow(unused_mut)]
        let mut dec: Arc<dyn TokenDecoder<T>> =
            Arc::new(SlabIndexDecoder::from_unified_vocab(self.vocab));

        #[cfg(feature = "rayon")]
        if self.parallel {
            dec = Arc::new(crate::utility::concurrency::rayon::ParallelRayonDecoder::new(dec));
        }

        dec
    }
}
