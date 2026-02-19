//! # Span Scanner

/// Trait for finding the next occurrence of a span.
pub trait SpanScanner: Send + Sync {
    /// Find the next occurrence of a span.
    fn next_span(
        &self,
        text: &str,
    ) -> Option<(usize, usize)>;
}
