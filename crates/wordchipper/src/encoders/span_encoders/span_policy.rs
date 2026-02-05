//! # Policy Trait for [`SpanEncoder`]s.

use crate::alloc::vec::Vec;
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;

/// The Span encoder trait for [`super::CompoundSpanVocabEncoder`].
pub trait SpanPolicy<T: TokenType>: Default {
    /// Encodes a single "word" span to (multiple?) tokens.
    ///
    /// ## Arguments
    /// * `vocab` - The unified token vocabulary reference..
    /// * `span` - The byte span to encode.
    /// * `tokens` - The target token buffer to append to.
    fn encode_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    );
}
