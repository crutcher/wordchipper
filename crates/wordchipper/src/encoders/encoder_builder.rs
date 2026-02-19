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
            .map(|(span, _)| String::from_utf8(span.clone()).unwrap())
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

    /// Load the full cl100k vocab from disk and compare logos vs regex
    /// on a long, realistic text. Run with:
    ///   cargo test -p wordchipper --release test_logos_vs_regex_full_vocab -- --nocapture --ignored
    #[test]
    #[ignore]
    fn test_logos_vs_regex_full_vocab() {
        type T = u32;
        let path = std::path::Path::new(env!("HOME"))
            .join("Library/Caches/wordchipper.io.crates.wordchipper/openai/cl100k_base/cl100k_base.tiktoken");
        if !path.exists() {
            eprintln!("Skipping: vocab not cached at {}", path.display());
            return;
        }

        let logos_vocab: UnifiedTokenVocab<T> = crate::pretrained::openai::OATokenizer::Cl100kBase
            .read_vocab(std::io::BufReader::new(std::fs::File::open(&path).unwrap()))
            .unwrap();

        let regex_spanning = logos_vocab.spanning().clone().with_pattern(
            crate::pretrained::openai::OATokenizer::Cl100kBase
                .factory()
                .pattern(),
        );
        let regex_vocab = UnifiedTokenVocab::<T>::new(
            regex_spanning,
            logos_vocab.span_vocab().clone(),
            logos_vocab.pair_vocab().clone(),
        )
        .unwrap();

        let logos_enc = TokenEncoderBuilder::new(logos_vocab.clone())
            .with_parallel(false)
            .init();
        let regex_enc = TokenEncoderBuilder::new(regex_vocab.clone())
            .with_parallel(false)
            .init();

        // Full text from benchmark mismatch (loaded from file if available).
        let mut texts: Vec<String> = vec![
            "Shakespeare's Macbeth: From Saga to Screen|\nA close reading of Shakespeare's play that will position the play in terms of its historical and political contexts and its relation to early modern discourses on the feminine, witchcraft, and the divinity of kings. We will begin with a consideration of the historical legends that constitute Shakespeare's \"sources,\" then read the play slowly and closely, coupling our discussions with readings from the period, exploring how Shakespeare's contemporaries thought of the political and cultural issues raised in the play.".into(),
            "'The items we buy are important, and we've been looking at 123 different options since 10:30am.".into(),
            "  Like all Choctaw counties, Wade County\u{2019}s boundaries were\n  modified before the Civil War\u{2014}in which the nation\u{2019}s capital".into(),
            "foo  \"bar\" baz\n   $400 (hello)\t\tworld".into(),
        ];
        // Try to load the actual failing benchmark text.
        let mismatch_path = "/tmp/wordchipper_mismatch.txt";
        if std::path::Path::new(mismatch_path).exists() {
            let full = std::fs::read_to_string(mismatch_path).unwrap();
            eprintln!("Loaded mismatch text: {} bytes", full.len());
            texts.push(full);
        }

        let logos_spanner =
            self::TokenEncoderBuilder::<T>::new(logos_vocab.clone()).build_spanner();
        let regex_spanner =
            self::TokenEncoderBuilder::<T>::new(regex_vocab.clone()).build_spanner();

        for text in &texts {
            // First compare spans to isolate the issue.
            let logos_spans = logos_spanner.split_spans(text);
            let regex_spans = regex_spanner.split_spans(text);
            if logos_spans != regex_spans {
                let div = logos_spans
                    .iter()
                    .zip(regex_spans.iter())
                    .position(|(a, e)| a != e)
                    .unwrap_or(logos_spans.len().min(regex_spans.len()));
                let lo = div.saturating_sub(2);
                let hi_l = (div + 3).min(logos_spans.len());
                let hi_r = (div + 3).min(regex_spans.len());
                let logos_words: Vec<&str> = logos_spans[lo..hi_l]
                    .iter()
                    .map(|s| {
                        let r = match s {
                            crate::spanning::SpanRef::Word(r)
                            | crate::spanning::SpanRef::Gap(r)
                            | crate::spanning::SpanRef::Special(r) => r.clone(),
                        };
                        &text[r]
                    })
                    .collect();
                let regex_words: Vec<&str> = regex_spans[lo..hi_r]
                    .iter()
                    .map(|s| {
                        let r = match s {
                            crate::spanning::SpanRef::Word(r)
                            | crate::spanning::SpanRef::Gap(r)
                            | crate::spanning::SpanRef::Special(r) => r.clone(),
                        };
                        &text[r]
                    })
                    .collect();
                panic!(
                    "SPAN mismatch at index {} (logos {} vs regex {} spans)\n\
                     logos[{}..{}]: {:?} = {:?}\n\
                     regex[{}..{}]: {:?} = {:?}\n\
                     text len: {}",
                    div,
                    logos_spans.len(),
                    regex_spans.len(),
                    lo,
                    hi_l,
                    &logos_spans[lo..hi_l],
                    logos_words,
                    lo,
                    hi_r,
                    &regex_spans[lo..hi_r],
                    regex_words,
                    text.len(),
                );
            }

            let logos_tokens = logos_enc.try_encode(text).unwrap();
            let regex_tokens = regex_enc.try_encode(text).unwrap();
            if logos_tokens != regex_tokens {
                let div = logos_tokens
                    .iter()
                    .zip(regex_tokens.iter())
                    .position(|(a, e)| a != e)
                    .unwrap_or(logos_tokens.len().min(regex_tokens.len()));
                panic!(
                    "Full-vocab token mismatch at index {} (logos {} vs regex {} tokens)\n\
                     logos[{}..{}]: {:?}\n\
                     regex[{}..{}]: {:?}\n\
                     text: {:?}",
                    div,
                    logos_tokens.len(),
                    regex_tokens.len(),
                    div.saturating_sub(3),
                    (div + 5).min(logos_tokens.len()),
                    &logos_tokens[div.saturating_sub(3)..(div + 5).min(logos_tokens.len())],
                    div.saturating_sub(3),
                    (div + 5).min(regex_tokens.len()),
                    &regex_tokens[div.saturating_sub(3)..(div + 5).min(regex_tokens.len())],
                    text,
                );
            }
        }
        eprintln!("All {} texts match!", texts.len());
    }

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
        ];

        for text in &samples {
            let regex_tokens = regex_enc.try_encode(text).unwrap();
            let logos_tokens = logos_enc.try_encode(text).unwrap();
            assert_eq!(regex_tokens, logos_tokens, "Token mismatch for: {text:?}");
        }
    }
}
