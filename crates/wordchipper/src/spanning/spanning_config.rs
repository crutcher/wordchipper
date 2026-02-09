//! # Text Spanner Configuration
use crate::regex::RegexWrapperPattern;
use crate::types::TokenType;
use crate::vocab::SpecialVocab;

/// Description of text spanning configuration.
///
/// ## Style Hints
///
/// Instance names should prefer `spanner_config`,
/// or `config` when there is no ambiguity.
#[derive(Debug, Clone)]
pub struct TextSpanningConfig<T: TokenType> {
    /// Regex pattern for word splitting.
    pub pattern: RegexWrapperPattern,

    /// Special tokens vocabulary.
    pub specials: SpecialVocab<T>,
}

impl<T: TokenType> From<RegexWrapperPattern> for TextSpanningConfig<T> {
    fn from(value: RegexWrapperPattern) -> Self {
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
        P: Into<RegexWrapperPattern>,
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
        P: Into<RegexWrapperPattern>,
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
    pub fn to_token_type<G: TokenType>(&self) -> anyhow::Result<TextSpanningConfig<G>> {
        Ok(TextSpanningConfig::<G> {
            pattern: self.pattern.clone(),
            specials: self.specials.to_token_type()?,
        })
    }

    /// Get the word pattern.
    pub fn pattern(&self) -> &RegexWrapperPattern {
        &self.pattern
    }

    /// Get the special tokens vocabulary.
    pub fn special_vocab(&self) -> &SpecialVocab<T> {
        &self.specials
    }

    /// Get a mutable view of the [`SpecialVocab`]
    pub fn special_vocab_mut(&mut self) -> &mut SpecialVocab<T> {
        &mut self.specials
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use crate::vocab::{SpecialVocab, TokenVocab};

    #[test]
    fn test_from_pattern() {
        type T = u32;

        let pattern = RegexWrapperPattern::Adaptive("hello".to_string());

        let mut config: TextSpanningConfig<T> = pattern.into();
        assert_eq!(config.pattern().as_str(), "hello");
        assert_eq!(config.special_vocab().len(), 0);

        config.special_vocab_mut().add_str_word("hello", 1);
        assert_eq!(config.special_vocab().len(), 1);

        let config = config.with_pattern("hi");
        assert_eq!(
            &config.pattern,
            &RegexWrapperPattern::Adaptive("hi".to_string())
        );

        let mut specials = SpecialVocab::default();
        specials.add_str_word("apple", 1);
        specials.add_str_word("pear", 1);

        let config = config.with_specials(specials.clone());
        assert_eq!(config.special_vocab(), &specials);
    }
}
