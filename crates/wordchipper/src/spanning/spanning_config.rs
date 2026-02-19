//! # Text Spanner Configuration
use crate::{
    regex::{ConstRegexPattern, RegexPattern},
    types::TokenType,
    vocab::SpecialVocab,
};

/// Selects which spanner implementation to use.
///
/// `Regex` uses runtime regex matching (supports arbitrary patterns).
/// `LogosCl100k` / `LogosO200k` use compile-time DFA lexers for known
/// `OpenAI` patterns, providing ~40-65x faster spanning.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SpannerPattern {
    /// Use regex-based spanning with the given pattern.
    Regex(RegexPattern),

    /// Use logos DFA for the `cl100k_base` pattern.
    LogosCl100k,

    /// Use logos DFA for the `o200k_base` pattern.
    LogosO200k,
}

impl From<RegexPattern> for SpannerPattern {
    fn from(pattern: RegexPattern) -> Self {
        Self::Regex(pattern)
    }
}

impl From<ConstRegexPattern> for SpannerPattern {
    fn from(pattern: ConstRegexPattern) -> Self {
        Self::Regex(pattern.into())
    }
}

impl<S: AsRef<str>> From<S> for SpannerPattern {
    fn from(pattern: S) -> Self {
        Self::Regex(RegexPattern::from(pattern))
    }
}

/// Description of text spanning configuration.
///
/// ## Style Hints
///
/// Instance names should prefer `spanner_config`,
/// or `config` when there is no ambiguity.
#[derive(Debug, Clone)]
pub struct TextSpanningConfig<T: TokenType> {
    /// Pattern / strategy for word splitting.
    pattern: SpannerPattern,

    /// Special tokens vocabulary.
    specials: SpecialVocab<T>,
}

impl<T: TokenType> From<RegexPattern> for TextSpanningConfig<T> {
    fn from(value: RegexPattern) -> Self {
        TextSpanningConfig::<T>::from_pattern(value)
    }
}

impl<T: TokenType> TextSpanningConfig<T> {
    /// Build a new config from the given word split pattern.
    ///
    /// Will contain an empty list of specials.
    ///
    /// ## Arguments
    /// * `pattern` - The word split pattern.
    pub fn from_pattern<P>(pattern: P) -> Self
    where
        P: Into<SpannerPattern>,
    {
        Self {
            pattern: pattern.into(),
            specials: SpecialVocab::default(),
        }
    }

    /// Set the word split pattern.
    ///
    /// ## Arguments
    /// * `pattern` - The new word split pattern.
    pub fn with_pattern<P>(
        self,
        pattern: P,
    ) -> Self
    where
        P: Into<SpannerPattern>,
    {
        Self {
            pattern: pattern.into(),
            ..self
        }
    }

    /// Set the special tokens vocabulary.
    ///
    /// ## Arguments
    /// * `specials` - The new special tokens vocabulary.
    pub fn with_specials<S>(
        self,
        specials: S,
    ) -> Self
    where
        S: Into<SpecialVocab<T>>,
    {
        let specials = specials.into();
        Self { specials, ..self }
    }

    /// Add the given special words.
    ///
    /// This does not replace existing special words.
    ///
    /// ## Arguments
    /// * `special_words` - An iterator of word strings and tokens.
    pub fn with_special_words<W, S>(
        self,
        special_words: W,
    ) -> Self
    where
        W: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
    {
        Self {
            specials: self.specials.with_special_words(special_words),
            ..self
        }
    }

    /// Convert to a different token type.
    pub fn to_token_type<G: TokenType>(&self) -> crate::errors::Result<TextSpanningConfig<G>> {
        Ok(TextSpanningConfig::<G> {
            pattern: self.pattern.clone(),
            specials: self.specials.to_token_type()?,
        })
    }

    /// Get the spanner pattern.
    pub fn pattern(&self) -> &SpannerPattern {
        &self.pattern
    }

    /// Get the special tokens vocabulary.
    pub fn specials(&self) -> &SpecialVocab<T> {
        &self.specials
    }

    /// Get a mutable view of the [`SpecialVocab`]
    pub fn specials_mut(&mut self) -> &mut SpecialVocab<T> {
        &mut self.specials
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::string::ToString,
        vocab::{SpecialVocab, VocabIndex},
    };

    #[test]
    fn test_from_pattern() {
        type T = u32;

        let pattern = RegexPattern::Adaptive("hello".to_string());

        let mut config: TextSpanningConfig<T> = pattern.into();
        assert_eq!(
            config.pattern(),
            &SpannerPattern::Regex(RegexPattern::Adaptive("hello".to_string()))
        );
        assert_eq!(config.specials().len(), 0);

        config.specials_mut().add_str_word("hello", 1);
        assert_eq!(config.specials().len(), 1);

        let config = config.with_pattern("hi");
        assert_eq!(
            &config.pattern,
            &SpannerPattern::Regex(RegexPattern::Adaptive("hi".to_string()))
        );

        let mut specials = SpecialVocab::default();
        specials.add_str_word("apple", 1);
        specials.add_str_word("pear", 1);

        let config = config.with_specials(specials.clone());
        assert_eq!(config.specials(), &specials);
    }

    #[test]
    fn test_logos_variants() {
        type T = u32;

        let config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(SpannerPattern::LogosCl100k);
        assert_eq!(config.pattern(), &SpannerPattern::LogosCl100k);

        let config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(SpannerPattern::LogosO200k);
        assert_eq!(config.pattern(), &SpannerPattern::LogosO200k);
    }
}
