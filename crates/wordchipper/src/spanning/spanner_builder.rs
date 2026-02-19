//! Text Spanner Builder

use core::num::NonZeroUsize;

use crate::{
    TokenType,
    alloc::sync::Arc,
    spanning::{RegexTextSpanner, TextSpanner, TextSpanningConfig},
};

/// Builder for [`TextSpanner`]s.
#[derive(Clone, PartialEq)]
pub struct TextSpannerBuilder<T: TokenType> {
    config: TextSpanningConfig<T>,

    parallel: bool,
    max_pool: Option<NonZeroUsize>,
}

impl<T: TokenType> TextSpannerBuilder<T> {
    /// Create a new [`TextSpannerBuilder`].
    ///
    /// Clones out the spanning configuration from the provided vocabulary.
    pub fn from_vocab(vocab: &crate::vocab::UnifiedTokenVocab<T>) -> Self {
        Self::new(vocab.spanning().clone())
    }

    /// Create a new [`TextSpannerBuilder`] with the given configuration.
    pub fn new(config: TextSpanningConfig<T>) -> Self {
        Self {
            config,
            parallel: true,
            max_pool: None,
        }
    }

    /// Get the underlying [`TextSpanningConfig`].
    pub fn config(&self) -> &TextSpanningConfig<T> {
        &self.config
    }

    /// Get whether the decoder should use parallel decoding.
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Set whether the decoder should use parallel decoding.
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.parallel = parallel;
    }

    /// Set whether the decoder should use parallel decoding.
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Get the max pool size for the [`TextSpanner`].
    pub fn max_pool(&self) -> Option<NonZeroUsize> {
        self.max_pool
    }

    /// Set the max pool size for the [`TextSpanner`].
    pub fn set_max_pool(
        &mut self,
        max_pool: NonZeroUsize,
    ) {
        self.max_pool = Some(max_pool);
    }

    /// Set the max pool size for the [`TextSpanner`].
    pub fn with_max_pool(
        mut self,
        max_pool: NonZeroUsize,
    ) -> Self {
        self.set_max_pool(max_pool);
        self
    }

    /// Build a [`TextSpanner`] with the current configuration.
    pub fn build(&self) -> Arc<dyn TextSpanner> {
        Arc::new(RegexTextSpanner::from_config(
            self.config.clone(),
            self.max_pool,
        ))
    }
}
