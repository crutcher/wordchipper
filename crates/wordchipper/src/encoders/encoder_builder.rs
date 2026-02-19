//! # `TokenEncoder` Builder

use core::num::NonZeroUsize;

use crate::{
    alloc::{boxed::Box, string::String, sync::Arc, vec::Vec},
    encoders::{
        TokenEncoder,
        span_encoders::{IncrementalSweepSpanEncoder, TokenSpanEncoder},
    },
    spanning::{LogosTextSpanner, RegexTextSpanner, SpannerPattern, TextSpanner},
    types::TokenType,
    vocab::{UnifiedTokenVocab, VocabIndex},
};

/// Builder for production [`TokenEncoder`]s.
pub struct TokenEncoderBuilder<T: TokenType> {
    vocab: UnifiedTokenVocab<T>,

    parallel: bool,
    max_pool: Option<NonZeroUsize>,
}

impl<T: TokenType> TokenEncoderBuilder<T> {
    /// Build a [`TokenEncoder`] with the default configuration.
    pub fn default(vocab: UnifiedTokenVocab<T>) -> Arc<dyn TokenEncoder<T>> {
        Self::new(vocab).init()
    }

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

    /// Collect the special token words from the vocab config.
    fn special_words(&self) -> Vec<String> {
        self.vocab
            .spanning()
            .specials()
            .span_pairs()
            .map(|(span, _)| {
                String::from_utf8(span.clone()).expect("special tokens must be valid UTF-8")
            })
            .collect()
    }

    /// Build the configured [`TextSpanner`], dispatching on [`SpannerPattern`].
    pub fn build_spanner(&self) -> Arc<dyn TextSpanner> {
        match self.vocab.spanning().pattern() {
            SpannerPattern::Regex(_) => Arc::new(RegexTextSpanner::from_config(
                self.vocab.spanning().clone(),
                self.max_pool,
            )),
            SpannerPattern::LogosCl100k => Arc::new(LogosTextSpanner::cl100k(self.special_words())),
            SpannerPattern::LogosO200k => Arc::new(LogosTextSpanner::o200k(self.special_words())),
        }
    }

    /// Build the configured `TokenEncoder`.
    pub fn init(self) -> Arc<dyn TokenEncoder<T>> {
        let spanner = self.build_spanner();

        #[allow(unused_mut)]
        let mut enc: Arc<dyn TokenEncoder<T>> = Arc::new(TokenSpanEncoder::<T>::new(
            spanner,
            self.vocab,
            Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<T>::default())),
        ));

        #[cfg(feature = "rayon")]
        if self.parallel {
            enc = Arc::new(crate::concurrency::rayon::ParallelRayonEncoder::new(enc));
        }

        enc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanning::{SpannerPattern, TextSpanningConfig},
        vocab::utility::testing::{build_test_shift_byte_vocab, build_test_vocab},
    };

    /// Build two encoders from the same BPE data but different spanners
    /// (regex vs logos) and verify they produce identical tokens.
    #[test]
    fn test_logos_vs_regex_encode_identical() {
        type T = u32;

        // Build vocab with regex-based spanning (the default factory path).
        let regex_config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let mut regex_vocab: UnifiedTokenVocab<T> =
            build_test_vocab(build_test_shift_byte_vocab(10), regex_config);
        let hi = regex_vocab.max_token().unwrap() + 1;
        regex_vocab.special_vocab_mut().add_str_word("<|HI|>", hi);

        // Clone BPE data, swap spanning config to logos.
        let logos_config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(SpannerPattern::LogosCl100k)
                .with_special_words([("<|HI|>", hi)]);
        let logos_vocab = UnifiedTokenVocab::<T>::new(
            logos_config,
            regex_vocab.span_vocab().clone(),
            regex_vocab.pair_vocab().clone(),
        )
        .unwrap();

        let regex_enc = TokenEncoderBuilder::new(regex_vocab)
            .with_parallel(false)
            .init();
        let logos_enc = TokenEncoderBuilder::new(logos_vocab)
            .with_parallel(false)
            .init();

        let samples = [
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
            "<|HI|>hello<|HI|>",
            "  multiple   spaces  ",
            "line1\nline2\r\nline3",
            "123 + 456 = 789",
            // Unicode letters and digits.
            "caf\u{00e9} na\u{00ef}ve \u{4f60}\u{597d}",
            "Geburtstag 2024: Alles Gute!",
            // Punctuation-heavy.
            "$$$!!!...---",
            // Edge cases.
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
