//! # Text Segmentation Configuration
use crate::regex::RegexWrapperPattern;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;

/// Word Split + Special Words Segmentor Configuration
#[derive(Debug, Clone)]
pub struct SegmentationConfig<T: TokenType> {
    /// Regex pattern for word splitting.
    pub pattern: RegexWrapperPattern,

    /// Special tokens vocabulary.
    pub specials: SpecialVocab<T>,
}

impl<T: TokenType> From<RegexWrapperPattern> for SegmentationConfig<T> {
    fn from(value: RegexWrapperPattern) -> Self {
        SegmentationConfig::<T>::from_pattern(value)
    }
}

impl<T: TokenType> SegmentationConfig<T> {
    /// Create a new text segmentor configuration with the given word pattern.
    ///
    /// Will contain an empty list of specials.
    ///
    /// ## Arguments
    /// * `pattern` - The word split pattern.
    ///
    /// ## Returns
    /// A new `SegmentationConfig` instance.
    pub fn from_pattern<P>(pattern: P) -> Self
    where
        P: Into<RegexWrapperPattern>,
    {
        Self {
            pattern: pattern.into(),
            specials: SpecialVocab::default(),
        }
    }

    /// Set the split pattern for the text segmentor configuration.
    ///
    /// ## Arguments
    /// * `pattern` - The new word split pattern.
    ///
    /// ## Returns
    /// The updated `SegmentationConfig` instance.
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

    /// Replace special tokens vocabulary.
    ///
    /// ## Arguments
    /// * `specials` - The new special tokens vocabulary.
    ///
    /// ## Returns
    /// The updated `SegmentationConfig` instance.
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

    /// Add all of the given special words to the specials.
    ///
    /// ## Arguments
    /// * `special_words` - An iterator of word strings and tokens.
    ///
    /// ## Returns
    /// The updated `SegmentationConfig` instance.
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

    /// Get the word pattern for the text segmentor configuration.
    ///
    /// ## Returns
    /// The regex pattern as a `String`.
    pub fn pattern(&self) -> &RegexWrapperPattern {
        &self.pattern
    }

    /// Get the special tokens vocabulary for the text segmentor configuration.
    ///
    /// ## Returns
    /// A reference to the internal `SpecialVocab`.
    pub fn special_vocab(&self) -> &SpecialVocab<T> {
        &self.specials
    }

    /// Get a mutable view of the [`SpecialVocab`]
    ///
    /// ## Returns
    /// A mutable reference to the internal `SpecialVocab`.
    pub fn special_vocab_mut(&mut self) -> &mut SpecialVocab<T> {
        &mut self.specials
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;

    #[test]
    fn test_from_pattern() {
        type T = u32;

        let pattern = RegexWrapperPattern::Adaptive("hello".to_string());

        let mut config: SegmentationConfig<T> = pattern.into();
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
