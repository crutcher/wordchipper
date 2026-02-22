//! Text Spanner Builder

use core::num::NonZeroUsize;

use crate::{
    TokenType,
    UnifiedTokenVocab,
    alloc::sync::Arc,
    spanning::{
        TextSpanner,
        TextSpanningConfig,
        span_lexers::{LexerTextSpanner, SpanLexer, build_regex_lexer},
    },
};

/// Builder for [`TextSpanner`]s.
#[derive(Clone, PartialEq)]
pub struct TextSpannerBuilder<T: TokenType> {
    config: TextSpanningConfig<T>,

    parallel: bool,
    max_pool: Option<NonZeroUsize>,
}

impl<T: TokenType> TextSpannerBuilder<T> {
    /// Build a new `Arc<dyn TextSpanner>` with defaults.
    pub fn default(vocab: &UnifiedTokenVocab<T>) -> Arc<dyn TextSpanner> {
        Self::from_vocab(vocab).build()
    }

    /// Create a new [`TextSpannerBuilder`].
    ///
    /// Clones out the spanning configuration from the provided vocabulary.
    pub fn from_vocab(vocab: &UnifiedTokenVocab<T>) -> Self {
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
    ///
    /// Automatically selects the fastest available word lexer for the
    /// configured pattern (e.g. a logos DFA accelerator if the `logos`
    /// feature is enabled and the pattern is recognized).
    /// Falls back to the compiled regex otherwise.
    /// The special lexer (if any) is always built from the regex pattern.
    pub fn build(&self) -> Arc<dyn TextSpanner> {
        let word_lexer: Arc<dyn SpanLexer> = build_regex_lexer(
            self.config().pattern().clone(),
            self.parallel,
            self.max_pool,
        );
        let special_lexer: Option<Arc<dyn SpanLexer>> = self
            .config
            .specials()
            .special_pattern()
            .map(|pattern| build_regex_lexer(pattern, self.parallel, self.max_pool));

        Arc::new(LexerTextSpanner::new(word_lexer, special_lexer))
    }
}
