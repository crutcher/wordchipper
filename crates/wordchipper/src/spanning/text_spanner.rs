//! # Text Spanner

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::compat::ranges::offset_range;
use crate::regex::{RegexWrapper, RegexWrapperPattern, alternate_choice_regex_pattern};
use crate::spanning::spanning_config::TextSpanningConfig;
use crate::types::TokenType;
use crate::vocab::TokenVocab;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use core::num::NonZeroUsize;
use core::ops::Range;

/// Span Label/Range Reference for [`TextSpanner`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpanRef {
    /// A normal word reference.
    Word(Range<usize>),

    /// A special word reference.
    Special(Range<usize>),

    /// A gap reference.
    Gap(Range<usize>),
}

impl From<SpanRef> for Range<usize> {
    fn from(span: SpanRef) -> Self {
        match span {
            SpanRef::Word(range) => range,
            SpanRef::Special(range) => range,
            SpanRef::Gap(range) => range,
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        use crate::concurrency::PoolToy;
        /// Text Spanner with Regex-based word splitting and special word matching.
        #[derive(Clone)]
        pub struct TextSpanner {
            /// Regex for splitting words.
            word_re: PoolToy<RegexWrapper>,

            /// Regex for matching special words.
            special_re: Option<PoolToy<RegexWrapper>>,
        }
    } else {
        /// Text Spanner with Regex-based word splitting and special word matching.
        #[derive(Clone)]
        pub struct TextSpanner {
            /// Regex for splitting words.
            word_re: RegexWrapper,

            /// Regex for matching special words.
            special_re: Option<RegexWrapper>,
        }
    }
}

impl TextSpanner {
    /// Build a new [`TextSpanner`] from a [`TextSpanningConfig`].
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
            .special_vocab()
            .span_pairs()
            .map(|(span, _)| String::from_utf8(span.clone()).unwrap())
            .collect::<Vec<_>>();

        Self::from_patterns(config.pattern, &specials, max_pool)
    }

    /// Build a new [`TextSpanner`] from patterns.
    ///
    /// ## Arguments
    /// * `word_pattern` - The word split pattern.
    /// * `specials` - A slice of special word strings.
    /// * `max_pool` - The maximum size of the regex pool; if None, lib defaults are used.
    pub fn from_patterns<P, S>(
        word_pattern: P,
        specials: &[S],
        max_pool: Option<NonZeroUsize>,
    ) -> Self
    where
        P: Into<RegexWrapperPattern>,
        S: AsRef<str>,
    {
        let span_re = word_pattern.into().into();

        let special_re = if specials.is_empty() {
            None
        } else {
            Some(alternate_choice_regex_pattern(specials).into())
        };

        Self::new(span_re, special_re, max_pool)
    }

    /// Build a new [`TextSpanner`] from regex.
    ///
    /// ## Arguments
    /// * `word_regex` - The regex for word splitting.
    /// * `special_regex` - The optional regex for special word matching.
    /// * `max_pool` - The maximum size of the regex pool; if None, lib defaults are used.
    pub fn new(
        word_re: RegexWrapper,
        special_re: Option<RegexWrapper>,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                use crate::concurrency::PoolToy;

                let word_re = PoolToy::new(word_re, max_pool);
                let special_re = special_re
                    .map(|r| PoolToy::new(r, max_pool));
            } else {
                let _ = max_pool;
            }
        }

        Self {
            word_re,
            special_re,
        }
    }

    /// Get the span split regex.
    pub fn word_regex(&self) -> &RegexWrapper {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                self.word_re.get()
            } else {
                &self.word_re
            }
        }
    }

    /// Get the optional special split regex.
    pub fn special_regex(&self) -> Option<&RegexWrapper> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                match self.special_re.as_ref() {
                    Some(p) => Some(p.get()),
                    None => None
                }
            } else {
                self.special_re.as_ref()
            }
        }
    }

    /// Find the next special span in the text.
    ///
    /// ## Arguments
    /// * `text` - The text to search in.
    ///
    /// ## Returns
    /// * `Some(Range<usize>)` if a special span is found,
    /// * `None` otherwise.
    pub fn next_special_span<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Option<Range<usize>> {
        match self.special_regex() {
            None => None,
            Some(re) => re.find_iter(text.as_ref()).next().map(|m| m.range()),
        }
    }

    /// Iterate over all split [`SpanRef`]s in the text.
    ///
    /// # Arguments
    /// * `text` - the text to split.
    /// * `f` - the function to apply to each span;
    ///   halts when the function returns `false`.
    ///
    /// Note: a byte is consumed *only if* the function returns `true`;
    /// if the function returns `false`, the byte is not consumed.
    ///
    /// # Returns
    /// ``(completed, consumed)`` where:
    /// - `consumed` is the number of bytes covered by spans accepted by `f`;
    /// - `completed` is if all spans were accepted.
    pub fn for_each_split_span<F>(
        &self,
        text: &str,
        f: &mut F,
    ) -> (bool, usize)
    where
        F: FnMut(SpanRef) -> bool,
    {
        let mut current = text;
        let mut offset = 0;

        while let Some(range) = self.next_special_span(current) {
            let Range { start, end } = range;
            let pre = &current[..start];

            let (cont, used) = self.for_each_word(pre, offset, f);
            if !cont {
                return (false, offset + used);
            }

            // we've consumed `offset + start` bytes at this point.
            if !f(SpanRef::Special(offset_range::<usize>(range, offset))) {
                // Special Exit
                return (false, offset + start);
            }

            // we've consumed `offset + end` bytes at this point.
            current = &current[end..];
            offset += end;
        }

        self.for_each_word(current, offset, f)
    }

    fn for_each_word<F>(
        &self,
        text: &str,
        offset: usize,
        f: &mut F,
    ) -> (bool, usize)
    where
        F: FnMut(SpanRef) -> bool,
    {
        let mut last = 0;
        for m in self.word_regex().find_iter(text) {
            let range = m.range();
            let Range { start, end } = range;

            if last < start {
                if !f(SpanRef::Gap(offset_range::<usize>(last..start, offset))) {
                    // Leading Gap Exit
                    return (false, last);
                }
                last = start;
            }

            if !f(SpanRef::Word(offset_range::<usize>(range, offset))) {
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

    /// Split text into spans.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    ///
    /// ## Returns
    /// A vector of `SpanRef` items.
    pub fn split_spans(
        &self,
        text: &str,
    ) -> Vec<SpanRef> {
        let capacity = text.len() as f32 / (EXPECTED_BYTES_PER_TOKEN * 0.8);
        let mut words = Vec::with_capacity(capacity as usize);

        self.for_each_split_span(text, &mut |span_ref| {
            words.push(span_ref);
            true
        });

        words
    }

    /// Rewrite text by splitting and re-joining without `Gap` matches.
    ///
    /// ## Arguments
    /// * `text` - The text to rewrite.
    ///
    /// ## Returns
    /// The rewritten string.
    pub fn remove_gaps<S: AsRef<str>>(
        &self,
        text: S,
    ) -> String {
        let text = text.as_ref();
        self.split_spans(text)
            .into_iter()
            .filter_map(|m| match m {
                SpanRef::Gap(_) => None,
                _ => Some(&text[Range::<usize>::from(m)]),
            })
            .collect()
    }

    /// Batch version of [`Self::remove_gaps`]
    pub fn batch_remove_gaps<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Vec<String> {
        texts.iter().map(|t| self.remove_gaps(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::vec;
    use crate::pretrained::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;

    #[test]
    fn test_for_each_split_span() {
        use SpanRef::*;
        type T = u32;

        let config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(r"\w+")
            .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let segmentor = TextSpanner::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

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
            TextSpanningConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN)
                .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let segmentor = TextSpanner::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

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

        let segmentor = TextSpanner::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

        let buf = vec!["hello world!", "abc def"];
        assert_eq!(
            segmentor.batch_remove_gaps(&buf),
            vec!["helloworld", "abcdef"]
        );
    }
}
