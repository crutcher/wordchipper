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

#[cfg(all(test, feature = "logos"))]
mod tests {
    use super::*;
    use crate::{
        VocabIndex,
        alloc::sync::Arc,
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        regex::RegexWrapper,
        spanning::{LexerTextSpanner, SpanLexer, TextSpanningConfig},
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
        let regex_enc: Arc<dyn crate::TokenEncoder<T>> =
            Arc::new(crate::encoders::span_encoders::TokenSpanEncoder::<T>::new(
                regex_spanner,
                vocab.clone(),
                Arc::new(|| {
                    Box::new(
                        crate::encoders::span_encoders::IncrementalSweepSpanEncoder::<T>::default(),
                    )
                }),
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
