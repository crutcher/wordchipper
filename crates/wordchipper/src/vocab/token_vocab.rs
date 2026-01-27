//! # Token Vocabulary Index

use crate::alloc::vec::Vec;
use crate::types::TokenType;

/// Common traits for token vocabularies.
pub trait TokenVocab<T: TokenType>: Clone + Send + Sync {
    /// Returns an iterator over all non-byte tokens.
    ///
    /// All returned tokens will have rank >= 256.
    ///
    /// ## Returns
    /// An iterator over the tokens.
    fn unordered_tokens(&self) -> impl Iterator<Item = T>;

    /// Returns a sorted vector of all tokens.
    ///
    /// ## Returns
    /// A sorted vector of tokens.
    fn sorted_tokens(&self) -> Vec<T> {
        let mut tokens: Vec<T> = self.unordered_tokens().collect();
        tokens.sort();
        tokens
    }

    /// Gets the highest ranked token.
    ///
    /// ## Returns
    /// The maximum token value.
    ///
    /// ## Panics
    /// Panics if the vocabulary is empty.
    fn max_token(&self) -> T {
        self.unordered_tokens().max().unwrap()
    }

    /// Generate all ``(Vec<u8>, T)`` pairs in the vocabulary.
    ///
    /// ## Returns
    /// An iterator over pairs of byte vectors and their corresponding tokens.
    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)>;
}
