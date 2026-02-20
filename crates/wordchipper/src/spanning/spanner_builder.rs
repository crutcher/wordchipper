//! Text Spanner Builder

use core::num::NonZeroUsize;

use cfg_if::cfg_if;

use crate::{
    TokenType,
    alloc::sync::Arc,
    regex::{RegexPattern, RegexWrapper},
    spanning::{LexerTextSpanner, SpanLexer, TextSpanner, TextSpanningConfig},
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
        fn maybe_pool(
            pattern: RegexPattern,
            max_pool: Option<NonZeroUsize>,
        ) -> Arc<dyn SpanLexer> {
            let re: RegexWrapper = pattern.into();

            cfg_if! {
                if #[cfg(feature = "std")] {
                    Arc::new(crate::concurrency::PoolToy::new(re, max_pool))
                } else {
                    let _ = max_pool;

                    Arc::new(re)
                }
            }
        }

        let word_lexer: Arc<dyn SpanLexer> = {
            let pattern = self.config().pattern().clone();
            maybe_pool(pattern, self.max_pool)
        };
        let special_lexer: Option<Arc<dyn SpanLexer>> = self
            .config
            .specials()
            .special_pattern()
            .map(|pattern| maybe_pool(pattern, self.max_pool));

        Arc::new(LexerTextSpanner::new(word_lexer, special_lexer))
    }
}
