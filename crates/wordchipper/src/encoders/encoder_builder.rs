//! # `TokenEncoder` Builder

use crate::{
    TokenType,
    alloc::sync::Arc,
    encoders::{
        TokenEncoder,
        token_span_encoder::{SpanEncoderSelector, TokenSpanEncoder},
    },
    spanning::TextSpannerBuilder,
    vocab::UnifiedTokenVocab,
};

/// Options for configuring a [`TokenEncoderBuilder`].
// TODO: serialize/deserialize?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenEncoderBuilderOptions {
    /// The [`UnifiedTokenVocab`] to use.
    ///
    /// When `None`, an appropriate default will be used for the concurrency.
    pub span_encoder: Option<SpanEncoderSelector>,

    /// Whether to use accelerated lexers.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub accelerated_lexers: bool,

    /// Should the encoder be threaded?
    pub parallel: bool,

    /// Should the encoder be concurrent?
    ///
    /// Concurrent encoders select defaults to be called concurrently.
    pub concurrent: bool,
}

impl Default for TokenEncoderBuilderOptions {
    fn default() -> Self {
        Self {
            span_encoder: None,
            accelerated_lexers: true,
            parallel: false,
            concurrent: false,
        }
    }
}

impl TokenEncoderBuilderOptions {
    /// Gets the effective span encoder selector.
    ///
    /// Will return any explict setting,
    /// otherwise will select based upon parallel and concurrency settings.
    pub fn effective_span_encoder(&self) -> SpanEncoderSelector {
        match self.span_encoder() {
            Some(selector) => selector,
            None if self.is_concurrent() => SpanEncoderSelector::ConcurrentDefault,
            _ => SpanEncoderSelector::SingleThreadDefault,
        }
    }

    /// Get the configured [`SpanEncoderSelector`].
    pub fn span_encoder(&self) -> Option<SpanEncoderSelector> {
        self.span_encoder
    }

    /// Set the configured [`SpanEncoderSelector`].
    pub fn set_span_encoder<E>(
        &mut self,
        span_encoder: E,
    ) where
        E: Into<Option<SpanEncoderSelector>>,
    {
        self.span_encoder = span_encoder.into();
    }

    /// Set the configured [`SpanEncoderSelector`] and return the builder.
    pub fn with_span_encoder<E>(
        mut self,
        span_encoder: E,
    ) -> Self
    where
        E: Into<Option<SpanEncoderSelector>>,
    {
        self.set_span_encoder(span_encoder);
        self
    }

    /// Are accelerated lexers enabled?
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub fn accelerated_lexers(&self) -> bool {
        self.accelerated_lexers
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub fn set_accelerated_lexers(
        &mut self,
        accelerated_lexers: bool,
    ) {
        self.accelerated_lexers = accelerated_lexers;
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub fn with_accelerated_lexers(
        mut self,
        accelerated_lexers: bool,
    ) -> Self {
        self.set_accelerated_lexers(accelerated_lexers);
        self
    }

    /// Gets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded encoder.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded encoder.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.parallel = parallel;
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded encoder.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Returns true if either parallel or concurrent is enabled.
    pub fn is_concurrent(&self) -> bool {
        self.concurrent || self.parallel
    }

    /// Gets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn concurrent(&self) -> bool {
        self.concurrent
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_concurrent(
        &mut self,
        concurrent: bool,
    ) {
        self.concurrent = concurrent;
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select a encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_concurrent(
        mut self,
        concurrent: bool,
    ) -> Self {
        self.set_concurrent(concurrent);
        self
    }
}

/// Builder for production [`TokenEncoder`]s.
///
/// The primary tuning knobs here are:
/// * [`set_parallel`](Self::set_parallel) - whether to request a parallel encoder (if supported).
/// * [`set_concurrent`](Self::set_concurrent) - whether to request a concurrent encoder (if supported).
///
/// See [`is_concurrent`](Self::is_concurrent) for more information on concurrent encoding.
#[derive(Clone, PartialEq)]
pub struct TokenEncoderBuilder<T: TokenType> {
    options: TokenEncoderBuilderOptions,
    vocab: Arc<UnifiedTokenVocab<T>>,
}

impl<T: TokenType> TokenEncoderBuilder<T> {
    /// Build a [`TokenEncoder`] with the default configuration.
    pub fn default(vocab: Arc<UnifiedTokenVocab<T>>) -> Arc<dyn TokenEncoder<T>> {
        Self::new(vocab).build()
    }

    /// Create a new builder for the vocab.
    pub fn new(vocab: Arc<UnifiedTokenVocab<T>>) -> Self {
        Self::new_with_options(vocab, Default::default())
    }

    /// Create a new builder for the vocab with the given options.
    pub fn new_with_options(
        vocab: Arc<UnifiedTokenVocab<T>>,
        options: TokenEncoderBuilderOptions,
    ) -> Self {
        Self { vocab, options }
    }

    /// Get the configured [`TokenEncoderBuilderOptions`].
    pub fn options(&self) -> TokenEncoderBuilderOptions {
        self.options
    }

    /// Get the underlying [`UnifiedTokenVocab`].
    pub fn vocab(&self) -> &Arc<UnifiedTokenVocab<T>> {
        &self.vocab
    }

    /// Gets the effective span encoder selector.
    ///
    /// Will return any explict setting,
    /// otherwise will select based upon parallel and concurrency settings.
    pub fn effective_span_encoder(&self) -> SpanEncoderSelector {
        self.options.effective_span_encoder()
    }

    /// Get the configured [`SpanEncoderSelector`].
    pub fn span_encoder(&self) -> Option<SpanEncoderSelector> {
        self.options.span_encoder()
    }

    /// Set the configured [`SpanEncoderSelector`].
    pub fn set_span_encoder<E>(
        &mut self,
        span_encoder: E,
    ) where
        E: Into<Option<SpanEncoderSelector>>,
    {
        self.options.set_span_encoder(span_encoder);
    }

    /// Set the configured [`SpanEncoderSelector`] and return the builder.
    pub fn with_span_encoder<E>(
        mut self,
        span_encoder: E,
    ) -> Self
    where
        E: Into<Option<SpanEncoderSelector>>,
    {
        self.set_span_encoder(span_encoder);
        self
    }

    /// Are accelerated lexers enabled?
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub fn accelerated_lexers(&self) -> bool {
        self.options.accelerated_lexers()
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub fn set_accelerated_lexers(
        &mut self,
        accelerated_lexers: bool,
    ) {
        self.options.set_accelerated_lexers(accelerated_lexers);
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanning.
    pub fn with_accelerated_lexers(
        mut self,
        accelerated_lexers: bool,
    ) -> Self {
        self.set_accelerated_lexers(accelerated_lexers);
        self
    }

    /// Gets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded encoder.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn parallel(&self) -> bool {
        self.options.parallel()
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded encoder.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.options.set_parallel(parallel);
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded encoder.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Returns true if either parallel or concurrent is enabled.
    pub fn is_concurrent(&self) -> bool {
        self.options.is_concurrent()
    }

    /// Gets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn concurrent(&self) -> bool {
        self.options.concurrent()
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_concurrent(
        &mut self,
        concurrent: bool,
    ) {
        self.options.set_concurrent(concurrent);
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select a encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_concurrent(
        mut self,
        concurrent: bool,
    ) -> Self {
        self.set_concurrent(concurrent);
        self
    }

    /// Build the configured `TokenEncoder`.
    pub fn build(&self) -> Arc<dyn TokenEncoder<T>> {
        let spanner = TextSpannerBuilder::from_vocab(self.vocab())
            .clone()
            .with_accelerated_lexers(self.accelerated_lexers())
            .with_concurrent(self.is_concurrent())
            .build();

        #[allow(unused_mut)]
        let mut enc: Arc<dyn TokenEncoder<T>> = Arc::new(TokenSpanEncoder::<T>::new_with_selector(
            spanner,
            self.vocab().clone(),
            self.effective_span_encoder(),
        ));

        #[cfg(feature = "rayon")]
        if self.parallel() {
            enc = Arc::new(crate::support::concurrency::rayon::ParallelRayonEncoder::new(enc));
        }

        enc
    }
}

#[cfg(all(test, feature = "logos"))]
mod tests {
    use super::*;
    use crate::{
        VocabIndex,
        alloc::sync::Arc,
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanning::{
            TextSpanningConfig,
            span_lexers::{LexerTextSpanner, SpanLexer},
        },
        support::regex::RegexWrapper,
        vocab::utility::testing::{build_test_shift_byte_vocab, build_test_vocab},
    };

    /// Build two encoders from the same BPE data but different spanners
    /// (regex vs logos auto-discovered) and verify they produce identical tokens.
    #[test]
    fn test_logos_vs_regex_encode_identical() {
        type T = u32;

        let config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let mut vocab: UnifiedTokenVocab<T> =
            build_test_vocab(build_test_shift_byte_vocab(10), config);
        let hi = vocab.max_token().unwrap() + 1;
        vocab.special_vocab_mut().add_str_word("<|HI|>", hi);

        let vocab = Arc::new(vocab);

        // Logos encoder: built via the normal path (auto-discovers logos DFA).
        let logos_enc = TokenEncoderBuilder::new(vocab.clone())
            .with_parallel(false)
            .build();

        // Regex encoder: manually construct a regex-only spanner.
        let regex_lexer: Arc<dyn SpanLexer> =
            Arc::new(RegexWrapper::from(vocab.spanning().pattern().clone()));
        let special_lexer: Option<Arc<dyn SpanLexer>> = vocab
            .spanning()
            .specials()
            .special_pattern()
            .map(|p| Arc::new(RegexWrapper::from(p)) as Arc<dyn SpanLexer>);
        let regex_spanner = Arc::new(LexerTextSpanner::new(regex_lexer, special_lexer));
        let regex_enc: Arc<dyn TokenEncoder<T>> =
            Arc::new(TokenSpanEncoder::<T>::new_with_selector(
                regex_spanner,
                vocab.clone(),
                SpanEncoderSelector::TailSweep,
            ));

        let samples = [
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
            "<|HI|>hello<|HI|>",
            "  multiple   spaces  ",
            "line1\nline2\r\nline3",
            "123 + 456 = 789",
            "caf\u{00e9} na\u{00ef}ve \u{4f60}\u{597d}",
            "Geburtstag 2024: Alles Gute!",
            "$$$!!!...---",
            "",
            " ",
            "a",
        ];

        for text in &samples {
            let regex_tokens = regex_enc.try_encode(text).unwrap();
            let logos_tokens = logos_enc.try_encode(text).unwrap();
            assert_eq!(regex_tokens, logos_tokens, "Token mismatch for: {text:?}");
        }
    }
}
