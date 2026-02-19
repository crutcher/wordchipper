//! # Span Lexer Trait

use crate::alloc::sync::Arc;

/// Trait for finding the next occurrence of a span.
pub trait SpanLexer: Send + Sync {
    /// Find the next occurrence of a span.
    ///
    /// ## Arguments
    /// * `text` - the text to scan over.
    /// * `offset` - the offset to start scanning from.
    ///
    /// ## Returns
    /// The span bounds, if found, relative to `text`.
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)>;
}

impl<T: SpanLexer> SpanLexer for Arc<T> {
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)> {
        self.as_ref().next_span(text, offset)
    }
}
