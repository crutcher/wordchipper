//! # `SpanLexer` trait

use core::ops::Deref;

/// Word-scanning plugin trait.
///
/// Implementors provide word-level text segmentation. The default
/// [`for_each_word`](Self::for_each_word) loops over
/// [`next_span`](Self::next_span) matches, emitting `Word` and `Gap` spans.
/// Lexers that produce richer token streams (e.g. logos DFA) override
/// `for_each_word` directly and leave `next_span` at its default.
///
/// ## Implementation Notes
///
/// Smart pointer types that implement `Deref<Target: SpanLexer>` (such as `Arc<T>`, `Box<T>`,
/// and [`PoolToy<T>`](crate::support::concurrency::PoolToy)) automatically implement `SpanLexer` through
/// a blanket implementation. This is the idiomatic Rust pattern used by the standard library
/// for traits like `Iterator` and `Future`.
pub trait SpanLexer: Send + Sync {
    /// Find the next match in `text` starting from `offset`.
    ///
    /// Returns `(start, end)` byte positions relative to `text`, or `None`.
    /// Used by the default [`for_each_word`](Self::for_each_word) and for
    /// special-token scanning. Implementations that override `for_each_word`
    /// can leave this at the default (returns `None`).
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)> {
        let _ = (text, offset);
        None
    }
}

// Blanket implementation for any type that derefs to a SpanLexer.
// This allows Arc<T>, Box<T>, PoolToy<T>, etc. to automatically implement SpanLexer.
impl<D> SpanLexer for D
where
    D: Deref + Send + Sync,
    D::Target: SpanLexer,
{
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)> {
        self.deref().next_span(text, offset)
    }
}
