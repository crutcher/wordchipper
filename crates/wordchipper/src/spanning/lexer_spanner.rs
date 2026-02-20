//! # Regex Text Spanner

use core::{num::NonZeroUsize, ops::Deref};

use crate::{
    alloc::{string::String, sync::Arc, vec::Vec},
    compat::ranges::offset_range,
    regex::{RegexPattern, RegexWrapper, alternate_choice_regex_pattern},
    spanning::{SpanRef, TextSpanner, TextSpanningConfig},
    types::TokenType,
    vocab::VocabIndex,
};

/// Trait for finding the next occurrence of a span.
///
/// ## Implementation Notes
///
/// Smart pointer types that implement `Deref<Target: SpanLexer>` (such as `Arc<T>`, `Box<T>`,
/// and [`PoolToy<T>`](crate::concurrency::PoolToy)) automatically implement `SpanLexer` through
/// a blanket implementation. This is the idiomatic Rust pattern used by the standard library
/// for traits like `Iterator` and `Future`.
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
/// Text Spanner with Regex-based word splitting and special word matching.
#[derive(Clone)]
pub struct LexerTextSpanner {
    word_scanner: Arc<dyn SpanLexer>,
    special_scanner: Option<Arc<dyn SpanLexer>>,
}

impl LexerTextSpanner {
    /// Build a new [`LexerTextSpanner`] from a [`TextSpanningConfig`].
    ///
    /// ## Arguments
    /// * `config` - The spanning configuration.
    pub fn from_config<T>(
        config: TextSpanningConfig<T>,

        max_pool: Option<NonZeroUsize>,
    ) -> Self
    where
        T: TokenType,
    {
        let specials = config
            .specials()
            .span_pairs()
            .map(|(span, _)| String::from_utf8(span.clone()).unwrap())
            .collect::<Vec<_>>();

        Self::from_patterns(config.pattern().clone(), &specials, max_pool)
    }

    /// Build a new [`LexerTextSpanner`] from patterns.
    ///
    /// ## Arguments
    /// * `word_pattern` - The word split pattern.
    /// * `specials` - A slice of special word strings.
    /// * `max_pool` - The maximum size of the regex pool; if None, lib defaults are used.
    pub fn from_patterns<P, S>(
        word_pattern: P,
        specials: &[S],
        _max_pool: Option<NonZeroUsize>,
    ) -> Self
    where
        P: Into<RegexPattern>,
        S: AsRef<str>,
    {
        let word_re: RegexWrapper = word_pattern.into().into();
        let word_re = Arc::new(word_re);

        let special_re: Option<Arc<dyn SpanLexer>> = if specials.is_empty() {
            None
        } else {
            let pattern = alternate_choice_regex_pattern(specials);
            let special_re: RegexWrapper = pattern.into();
            Some(Arc::new(special_re))
        };

        Self::new(word_re, special_re)
    }

    /// Build a new [`LexerTextSpanner`] from regex.
    ///
    /// ## Arguments
    /// * `word_scanner` - The regex for word splitting.
    /// * `special_scanner` - The optional regex for special word matching.
    pub fn new(
        word_scanner: Arc<dyn SpanLexer>,
        special_scanner: Option<Arc<dyn SpanLexer>>,
    ) -> Self {
        Self {
            word_scanner,
            special_scanner,
        }
    }

    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        let mut last = 0;
        while let Some((start, end)) = self.word_scanner.next_span(text, last) {
            if last < start {
                if !f(SpanRef::Gap(offset_range::<usize>(last..start, offset))) {
                    // Leading Gap Exit
                    return (false, last);
                }
                last = start;
            }

            if !f(SpanRef::Word(offset_range::<usize>(start..end, offset))) {
                // Word Exit
                return (false, last);
            }
            last = end;
        }

        if last < text.len() {
            if !f(SpanRef::Gap(offset_range::<usize>(
                last..text.len(),
                offset,
            ))) {
                // Trailing Gap Exit
                return (false, last);
            }
            last = text.len();
        }

        (true, last)
    }

    fn next_special_span(
        &self,
        text: &str,
    ) -> Option<(usize, usize)> {
        match &self.special_scanner {
            None => None,
            Some(scanner) => scanner.next_span(text, 0),
        }
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

            let (cont, used) = self.for_each_word(pre, offset, f);
            if !cont {
                return (false, offset + used);
            }

            // we've consumed `offset + start` bytes at this point.
            if !f(SpanRef::Special(offset_range::<usize>(start..end, offset))) {
                // Special Exit
                return (false, offset + start);
            }

            // we've consumed `offset + end` bytes at this point.
            current = &current[end..];
            offset += end;
        }

        self.for_each_word(current, offset, f)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        alloc::{boxed::Box, vec},
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanning::{SpanRef, TextSpanningConfig},
    };

    #[test]
    fn test_box() {
        let config: TextSpanningConfig<u32> = TextSpanningConfig::from_pattern(r"\w+");
        let _box: Box<dyn TextSpanner> = Box::new(LexerTextSpanner::from_config(
            config,
            Some(NonZeroUsize::new(1).unwrap()),
        ));
    }

    #[test]
    fn test_for_each_split_span() {
        use crate::spanning::text_spanner::SpanRef::*;
        type T = u32;

        let config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(r"\w+")
            .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let segmentor = LexerTextSpanner::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

        let source = "abc 1<|FNORD|> def  <|NORP|> ghi   ";

        let mut spans: Vec<SpanRef> = Vec::new();
        segmentor.for_each_split_span(source, &mut |span_ref| {
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
        segmentor.for_each_split_span("   abc", &mut |span_ref| match span_ref {
            Word(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Gap(0..3)]);

        // Test "for_each_split_span" Special Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        segmentor.for_each_split_span("abc   def<|FNORD|>", &mut |span_ref| match span_ref {
            Special(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3), Gap(3..6), Word(6..9)]);

        // Test "for_each_word" Leading Gap Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        segmentor.for_each_split_span("abc  def", &mut |span_ref| match span_ref {
            Gap(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3)]);

        // Test "for_each_word" Trailing Gap Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        segmentor.for_each_split_span("foo  ", &mut |span_ref| match span_ref {
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

        let segmentor = LexerTextSpanner::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

        let buf = "hello<|FNORD|> wor<|NORP|>ld!";

        assert_eq!(
            &segmentor.split_spans(buf),
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

        let segmentor = LexerTextSpanner::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

        let buf = vec!["hello world!", "abc def"];
        assert_eq!(
            segmentor.batch_remove_gaps(&buf),
            vec!["helloworld", "abcdef"]
        );
    }
}
