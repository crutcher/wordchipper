//! # `TokenEncoder` Builder

use core::num::NonZeroUsize;

use crate::{
    alloc::{boxed::Box, sync::Arc},
    encoders::{
        TokenEncoder,
        span_encoders::{IncrementalSweepSpanEncoder, TokenSpanEncoder},
    },
    spanning::TextSpannerBuilder,
    types::TokenType,
    vocab::UnifiedTokenVocab,
};

/// Builder for production [`TokenEncoder`]s.
#[derive(Clone, PartialEq)]
pub struct TokenEncoderBuilder<T: TokenType> {
    vocab: UnifiedTokenVocab<T>,
    spanner_builder: TextSpannerBuilder<T>,
}

impl<T: TokenType> TokenEncoderBuilder<T> {
    /// Build a [`TokenEncoder`] with the default configuration.
    pub fn default(vocab: UnifiedTokenVocab<T>) -> Arc<dyn TokenEncoder<T>> {
        Self::new(vocab).build()
    }

    /// Create a new builder for the vocab.
    pub fn new(vocab: UnifiedTokenVocab<T>) -> Self {
        let spanner_builder = TextSpannerBuilder::from_vocab(&vocab);
        Self {
            vocab,
            spanner_builder,
        }
    }

    /// Get the underlying [`UnifiedTokenVocab`].
    pub fn vocab(&self) -> &UnifiedTokenVocab<T> {
        &self.vocab
    }

    /// Get the underlying [`TextSpannerBuilder`].
    pub fn spanner_builder(&self) -> &TextSpannerBuilder<T> {
        &self.spanner_builder
    }

    /// Get the underlying [`TextSpannerBuilder`] for mutable access.
    pub fn spanner_builder_mut(&mut self) -> &mut TextSpannerBuilder<T> {
        &mut self.spanner_builder
    }

    /// Get whether the decoder should use parallel decoding.
    pub fn parallel(&self) -> bool {
        self.spanner_builder.parallel()
    }

    /// Set whether the decoder should use parallel decoding.
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.spanner_builder_mut().set_parallel(parallel);
    }

    /// Set whether the decoder should use parallel decoding.
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Get the max parallel configured pool size.
    ///
    /// If none, will use system and environment defaults.
    pub fn max_pool(&self) -> Option<NonZeroUsize> {
        self.spanner_builder.max_pool()
    }

    /// Set the max parallel configured pool size.
    ///
    /// If none, will use system and environment defaults.
    pub fn set_max_pool(
        &mut self,
        max_pool: NonZeroUsize,
    ) {
        self.spanner_builder_mut().set_max_pool(max_pool);
    }

    /// Set the max parallel configured pool size.
    ///
    /// If none, will use system and environment defaults.
    pub fn with_max_pool(
        mut self,
        max_pool: NonZeroUsize,
    ) -> Self {
        self.set_max_pool(max_pool);
        self
    }

    /// Build the configured `TokenEncoder`.
    pub fn build(&self) -> Arc<dyn TokenEncoder<T>> {
        let spanner = self.spanner_builder().build();

        #[allow(unused_mut)]
        let mut enc: Arc<dyn TokenEncoder<T>> = Arc::new(TokenSpanEncoder::<T>::new(
            spanner,
            self.vocab().clone(),
            Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<T>::default())),
        ));

        #[cfg(feature = "rayon")]
        if self.parallel() {
            enc = Arc::new(crate::concurrency::rayon::ParallelRayonEncoder::new(enc));
        }

        enc
    }
}
