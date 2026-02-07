//! # Token Vocabulary Index

use crate::alloc::vec::Vec;
use crate::types::TokenType;

/// Common traits for token vocabularies.
pub trait TokenVocab<T: TokenType>: Clone + Send + Sync {
    /// The token type: T.
    type Token: TokenType;

    /// Returns a vector of all tokens, sorted.
    fn tokens(&self) -> Vec<T>;

    /// Returns the number of tokens in the vocabulary.
    fn len(&self) -> usize {
        self.tokens().len()
    }

    /// Returns true if the vocabulary is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the highest ranked token.
    ///
    /// ## Returns
    /// The maximum token value, or None.
    fn max_token(&self) -> Option<T> {
        self.tokens().last().copied()
    }

    /// Generate all ``(Vec<u8>, T)`` pairs in the vocabulary.
    ///
    /// ## Returns
    /// An iterator over pairs of byte vectors and their corresponding tokens.
    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)>;
}
