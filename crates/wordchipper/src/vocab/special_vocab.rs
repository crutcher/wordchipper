//! # Special Words Vocabulary

use crate::alloc::vec::Vec;
use crate::types::{SpanTokenMap, TokenType};
use crate::vocab::TokenVocab;

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
///
/// This contains no byte:token mappings, or pair mergers.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct SpecialVocab<T: TokenType> {
    /// The regex pattern used for text spl
    /// Map of ``{ Vec<u8> -> T }``.
    span_map: SpanTokenMap<T>,
}

impl<T: TokenType> From<SpanTokenMap<T>> for SpecialVocab<T> {
    fn from(span_map: SpanTokenMap<T>) -> Self {
        Self::new(span_map)
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
    pub fn new(span_map: SpanTokenMap<T>) -> Self {
        Self { span_map }
    }

    /// Get the length of the special words vocab.
    ///
    /// ## Returns
    /// The number of entries in the special vocabulary.
    pub fn len(&self) -> usize {
        self.span_map.len()
    }

    /// Returns `true` if the special words vocab contains no words.
    ///
    /// ## Returns
    /// `true` if the special vocabulary is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.span_map.is_empty()
    }

    /// Get the span => token map.
    ///
    /// ## Returns
    /// A reference to the internal `SpanTokenMap`.
    pub fn span_map(&self) -> &SpanTokenMap<T> {
        &self.span_map
    }

    /// Get an iterator over the span slices.
    ///
    /// ## Returns
    /// An iterator over the byte slices of special words.
    pub fn spans(&self) -> impl Iterator<Item = &[u8]> {
        self.span_map.keys().map(|chunk| chunk.as_slice())
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
    fn unordered_tokens(&self) -> impl Iterator<Item = T> {
        self.span_map.values().copied()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_map
            .iter()
            .map(|(chunk, &token)| (chunk.clone(), token))
    }
}
