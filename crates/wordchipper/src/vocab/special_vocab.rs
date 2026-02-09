//! # Special Words Vocabulary

use crate::alloc::vec::Vec;
use crate::types::{CommonHashSet, TokenType};
use crate::vocab::utility::validators::try_vocab_size;
use crate::vocab::{SpanTokenMap, TokenVocab};

/// Token vocabulary for special words.
///
/// This contains no byte:token mappings, or pair mergers.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct SpecialVocab<T: TokenType> {
    /// The map of special words to tokens.
    span_map: SpanTokenMap<T>,
}

impl<T: TokenType> From<SpanTokenMap<T>> for SpecialVocab<T> {
    fn from(span_map: SpanTokenMap<T>) -> Self {
        Self::from_map(span_map)
    }
}

impl<T: TokenType> SpecialVocab<T> {
    /// Create a new special words vocab.
    ///
    /// ## Arguments
    /// * `span_map` - A mapping of byte spans to tokens.
    ///
    /// ## Returns
    /// A new `SpecialVocab` instance.
    pub fn from_map(span_map: SpanTokenMap<T>) -> Self {
        Self { span_map }
    }

    /// Get the span map.
    pub fn span_map(&self) -> &SpanTokenMap<T> {
        &self.span_map
    }

    /// Convert to a different token type.
    pub fn to_token_type<G: TokenType>(&self) -> anyhow::Result<SpecialVocab<G>> {
        if let Some(max) = self.max_token() {
            try_vocab_size::<G>(max.to_usize().unwrap())?;
        }

        Ok(SpecialVocab::<G>::from_map(
            self.span_map
                .iter()
                .map(|(chunk, &token)| (chunk.clone(), G::from(token).unwrap()))
                .collect(),
        ))
    }

    /// Add a word to the vocab.
    ///
    /// ## Arguments
    /// * `word` - The word string to add.
    /// * `token` - The token value to assign to the word.
    pub fn add_str_word(
        &mut self,
        word: &str,
        token: T,
    ) {
        self.span_map.insert(word.as_bytes().to_vec(), token);
    }

    /// Extend the vocabulary with the given special words.
    ///
    /// ## Arguments
    /// * `special_words` - An iterator of word strings and tokens.
    ///
    /// ## Returns
    /// The updated `SpecialVocab` instance.
    pub fn with_special_words<W, S>(
        self,
        special_words: W,
    ) -> Self
    where
        W: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
    {
        let mut vocab = self;
        for (word, token) in special_words {
            vocab.add_str_word(word.as_ref(), token);
        }
        vocab
    }

    /// Return the associated token for the word, if any.
    ///
    /// ## Arguments
    /// * `chunk` - The byte slice to look up.
    ///
    /// ## Returns
    /// An `Option` containing the token if the span exists in the special vocabulary.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        self.span_map.get(chunk).copied()
    }
}

impl<T: TokenType> TokenVocab<T> for SpecialVocab<T> {
    type Token = T;
    fn len(&self) -> usize {
        self.span_map.len()
    }

    fn tokens(&self) -> CommonHashSet<T> {
        self.span_map.values().copied().collect()
    }

    fn max_token(&self) -> Option<T> {
        self.span_map.values().max().copied()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_map
            .iter()
            .map(|(chunk, &token)| (chunk.clone(), token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_vocab() {
        type T = u32;
        let mut vocab: SpecialVocab<T> = SpecialVocab::default();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);

        vocab.add_str_word("hello", 1);
        assert_eq!(vocab.len(), 1);
        assert!(!vocab.is_empty());

        assert_eq!(
            &vocab.span_map,
            &[("hello".as_bytes().to_vec(), 1)].into_iter().collect()
        );

        let rebuild: SpecialVocab<T> = vocab.span_map.clone().into();
        assert_eq!(rebuild, vocab);
    }
}
