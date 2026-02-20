//! # Text Spanner Configuration
use crate::{
    alloc::sync::Arc,
    regex::RegexPattern,
    spanning::SpanLexer,
    types::TokenType,
    vocab::SpecialVocab,
};

/// Description of text spanning configuration.
///
/// ## Style Hints
///
/// Instance names should prefer `spanner_config`,
/// or `config` when there is no ambiguity.
#[derive(Debug, Clone)]
pub struct TextSpanningConfig<T: TokenType> {
    /// Regex pattern for word splitting.
    pattern: RegexPattern,

    /// Special tokens vocabulary.
    specials: SpecialVocab<T>,

    /// Optional factory for the word lexer.
    /// When set, `TextSpannerBuilder::build()` uses this instead of
    /// compiling the regex pattern into a word lexer.
    word_lexer_factory: Option<fn() -> Arc<dyn SpanLexer>>,
}

// Manual PartialEq: fn pointers trigger `unpredictable_function_pointer_comparisons`,
// and the factory isn't meaningful for config equality anyway.
impl<T: TokenType> PartialEq for TextSpanningConfig<T> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.pattern == other.pattern && self.specials == other.specials
    }
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
        P: Into<RegexPattern>,
    {
        Self {
            pattern: pattern.into(),
            specials: SpecialVocab::default(),
            word_lexer_factory: None,
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
        P: Into<RegexPattern>,
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
            word_lexer_factory: self.word_lexer_factory,
        })
    }

    /// Get the word pattern.
    pub fn pattern(&self) -> &RegexPattern {
        &self.pattern
    }

    /// Get the special tokens vocabulary.
    pub fn specials(&self) -> &SpecialVocab<T> {
        &self.specials
    }

    /// Get the special pattern, if any.
    pub fn special_pattern(&self) -> Option<RegexPattern> {
        self.specials.special_pattern()
    }

    /// Get a mutable view of the [`SpecialVocab`]
    pub fn specials_mut(&mut self) -> &mut SpecialVocab<T> {
        &mut self.specials
    }

    /// Get the word lexer factory, if set.
    pub fn word_lexer_factory(&self) -> Option<fn() -> Arc<dyn SpanLexer>> {
        self.word_lexer_factory
    }

    /// Set a factory for the word lexer, overriding the regex-compiled default.
    pub fn with_word_lexer_factory(
        self,
        factory: fn() -> Arc<dyn SpanLexer>,
    ) -> Self {
        Self {
            word_lexer_factory: Some(factory),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{alloc::string::ToString, vocab::SpecialVocab};

    #[test]
    fn test_from_pattern() {
        type T = u32;

        let pattern = RegexPattern::Adaptive("hello".to_string());

        let mut config: TextSpanningConfig<T> = pattern.into();
        assert_eq!(config.pattern().as_str(), "hello");
        assert_eq!(config.specials().len(), 0);

        config.specials_mut().add_str_word("hello", 1);
        assert_eq!(config.specials().len(), 1);

        let config = config.with_pattern("hi");
        assert_eq!(&config.pattern, &RegexPattern::Adaptive("hi".to_string()));

        let mut specials = SpecialVocab::default();
        specials.add_str_word("apple", 1);
        specials.add_str_word("pear", 1);

        let config = config.with_specials(specials.clone());
        assert_eq!(config.specials(), &specials);
    }
}
