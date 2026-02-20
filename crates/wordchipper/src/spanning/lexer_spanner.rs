//! # Lexer Text Spanner

use core::ops::Deref;

use crate::{
    alloc::sync::Arc,
    compat::ranges::offset_range,
    spanning::{SpanRef, TextSpanner},
};

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
/// and [`PoolToy<T>`](crate::concurrency::PoolToy)) automatically implement `SpanLexer` through
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

    /// Scan `text` into [`Word`](SpanRef::Word) and [`Gap`](SpanRef::Gap) spans.
    ///
    /// The default implementation loops over [`next_span`](Self::next_span),
    /// classifying matched regions as `Word` and unmatched regions as `Gap`.
    /// Lexers with richer token classification override this directly.
    ///
    /// ## Arguments
    /// * `text` - the text segment to scan (no special tokens).
    /// * `offset` - byte offset to add to emitted span ranges.
    /// * `f` - callback; return `false` to halt early.
    ///
    /// ## Returns
    /// `(completed, consumed)` where `consumed` is the byte count of
    /// accepted spans and `completed` indicates all spans were accepted.
    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        let mut last = 0;
        while let Some((start, end)) = self.next_span(text, last) {
            if last < start {
                if !f(SpanRef::Gap(offset_range::<usize>(last..start, offset))) {
                    return (false, last);
                }
                last = start;
            }

            if !f(SpanRef::Word(offset_range::<usize>(start..end, offset))) {
                return (false, last);
            }
            last = end;
        }

        if last < text.len() {
            if !f(SpanRef::Gap(offset_range::<usize>(
                last..text.len(),
                offset,
            ))) {
                return (false, last);
            }
            last = text.len();
        }

        (true, last)
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

    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        self.deref().for_each_word(text, offset, f)
    }
}

/// A [`TextSpanner`] composed over [`SpanLexer`] plugins.
///
/// Combines a word-scanning [`SpanLexer`] with an optional special-token
/// scanner. The word lexer handles segmentation within text segments;
/// the special lexer finds special tokens that split the input into
/// those segments.
///
/// The word lexer is pluggable (e.g. regex-based or logos DFA). The special
/// lexer is always regex-based, built from the special token patterns.
#[derive(Clone)]
pub struct LexerTextSpanner {
    word_lexer: Arc<dyn SpanLexer>,
    special_lexer: Option<Arc<dyn SpanLexer>>,
}

impl LexerTextSpanner {
    /// Build a new [`LexerTextSpanner`].
    ///
    /// ## Arguments
    /// * `word_scanner` - The lexer for word splitting.
    /// * `special_scanner` - The optional lexer for special word matching.
    pub fn new(
        word_scanner: Arc<dyn SpanLexer>,
        special_scanner: Option<Arc<dyn SpanLexer>>,
    ) -> Self {
        Self {
            word_lexer: word_scanner,
            special_lexer: special_scanner,
        }
    }

    fn next_special_span(
        &self,
        text: &str,
    ) -> Option<(usize, usize)> {
        self.special_lexer
            .as_ref()
            .and_then(|s| s.next_span(text, 0))
    }
}

impl TextSpanner for LexerTextSpanner {
    fn for_each_split_span(
        &self,
        text: &str,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        let mut current = text;
        let mut offset = 0;

        while let Some((start, end)) = self.next_special_span(current) {
            let pre = &current[..start];

            let (cont, used) = self.word_lexer.for_each_word(pre, offset, f);
            if !cont {
                return (false, offset + used);
            }

            if !f(SpanRef::Special(offset_range::<usize>(start..end, offset))) {
                return (false, offset + start);
            }

            current = &current[end..];
            offset += end;
        }

        self.word_lexer.for_each_word(current, offset, f)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        TokenType,
        alloc::{boxed::Box, vec, vec::Vec},
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanning::{SpanRef, TextSpanningConfig},
    };

    const _LEXER_SPANNER_BOX_CHECK: Option<Box<LexerTextSpanner>> = None;
    const _LEXER_SPANNER_ARC_CHECK: Option<Arc<LexerTextSpanner>> = None;

    fn from_config<T: TokenType>(config: &TextSpanningConfig<T>) -> LexerTextSpanner {
        LexerTextSpanner::new(
            Arc::new(config.pattern().clone().compile().unwrap()),
            config
                .special_pattern()
                .map(|p| Arc::new(p.compile().unwrap()) as Arc<dyn SpanLexer>),
        )
    }

    #[test]
    fn test_for_each_split_span() {
        use crate::spanning::text_spanner::SpanRef::*;
        type T = u32;

        let config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(r"\w+")
            .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let spanner = from_config(&config);

        let source = "abc 1<|FNORD|> def  <|NORP|> ghi   ";

        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span(source, &mut |span_ref| {
            spans.push(span_ref);
            true
        });
        assert_eq!(
            spans,
            vec![
                Word(0..3),
                Gap(3..4),
                Word(4..5),
                Special(5..14),
                Gap(14..15),
                Word(15..18),
                Gap(18..20),
                Special(20..28),
                Gap(28..29),
                Word(29..32),
                Gap(32..35),
            ]
        );

        // The following are white-box tests to exercise the different halting points.

        // Test "for_each_split_span" Word Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("   abc", &mut |span_ref| match span_ref {
            Word(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Gap(0..3)]);

        // Test "for_each_split_span" Special Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("abc   def<|FNORD|>", &mut |span_ref| match span_ref {
            Special(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3), Gap(3..6), Word(6..9)]);

        // Test "for_each_word" Leading Gap Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("abc  def", &mut |span_ref| match span_ref {
            Gap(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3)]);

        // Test "for_each_word" Trailing Gap Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("foo  ", &mut |span_ref| match span_ref {
            Gap(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3)]);
    }

    #[test]
    fn test_split_words() {
        type T = u32;

        let config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN)
                .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let spanner = from_config(&config);

        let buf = "hello<|FNORD|> wor<|NORP|>ld!";

        assert_eq!(
            &spanner.split_spans(buf),
            &vec![
                SpanRef::Word(0..5),
                SpanRef::Special(5..14),
                SpanRef::Word(14..18),
                SpanRef::Special(18..26),
                SpanRef::Word(26..28),
                SpanRef::Word(28..buf.len()),
            ]
        );
    }

    #[test]
    fn test_rewrite() {
        type T = u32;

        let config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(r"\w+");

        let spanner = from_config(&config);

        let buf = vec!["hello world!", "abc def"];
        assert_eq!(
            spanner.batch_remove_gaps(&buf),
            vec!["helloworld", "abcdef"]
        );
    }
}
