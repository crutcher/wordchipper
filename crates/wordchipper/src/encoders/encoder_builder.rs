//! # `TokenEncoder` Builder

use core::num::NonZeroUsize;

use crate::{
    alloc::sync::Arc,
    encoders::{TokenEncoder, span_encoders::CompoundSpanVocabEncoder},
    spanning::{RegexTextSpanner, TextSpanner},
    types::TokenType,
    vocab::UnifiedTokenVocab,
};

/// Builder for configuring a [`TokenEncoder`].
pub struct TokenEncoderBuilder<T: TokenType> {
    vocab: UnifiedTokenVocab<T>,

    parallel: bool,
    max_pool: Option<NonZeroUsize>,
}

impl<T: TokenType> TokenEncoderBuilder<T> {
    /// Create a new builder for the vocab.
    pub fn new(vocab: UnifiedTokenVocab<T>) -> Self {
        Self {
            vocab,
            max_pool: None,
            parallel: true,
        }
    }

    /// Get the underlying [`UnifiedTokenVocab`].
    pub fn vocab(&self) -> &UnifiedTokenVocab<T> {
        &self.vocab
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

    /// Get the max pool size for the [`RegexTextSpanner`].
    pub fn max_pool(&self) -> Option<NonZeroUsize> {
        self.max_pool
    }

    /// Set the max pool size for the [`RegexTextSpanner`].
    pub fn with_max_pool(
        mut self,
        max_pool: NonZeroUsize,
    ) -> Self {
        self.max_pool = Some(max_pool);
        self
    }

    /// Build the configured [`RegexTextSpanner`].
    pub fn build_spanner(&self) -> Arc<dyn TextSpanner> {
        Arc::new(RegexTextSpanner::from_config(
            self.vocab.spanning().clone(),
            self.max_pool,
        ))
    }

    /// Build the configured `TokenEncoder`.
    pub fn init(&self) -> Arc<dyn TokenEncoder<T>> {
        let spanner = self.build_spanner();

        #[allow(unused_mut)]
        let mut enc: Arc<dyn TokenEncoder<T>> = Arc::new(CompoundSpanVocabEncoder::<T>::new(
            spanner,
            self.vocab.clone(),
        ));

        #[cfg(feature = "rayon")]
        if self.parallel {
            enc = Arc::new(crate::concurrency::rayon::ParallelRayonEncoder::new(enc));
        }

        enc
    }
}
