//! # Text Segmentor

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::concurrency::pool_toy::PoolToy;
use crate::regex::exact_match_union::exact_match_union_regex_pattern;
use crate::regex::{RegexWrapper, RegexWrapperPattern};
use crate::segmentation::segmentation_config::SegmentationConfig;
use crate::types::TokenType;
use crate::vocab::TokenVocab;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use core::ops::Range;
use std::num::NonZeroUsize;

/// Word Reference for [`TextSegmentor`].
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

fn offset_range(
    range: Range<usize>,
    offset: usize,
) -> Range<usize> {
    Range {
        start: range.start + offset,
        end: range.end + offset,
    }
}

/// Word Split + Special Words Segmentor
#[derive(Clone)]
pub struct TextSegmentor {
    /// Regex for splitting words.
    pub word_re_pool: PoolToy<RegexWrapper>,

    /// Regex for matching special words.
    pub special_re_pool: Option<PoolToy<RegexWrapper>>,
}

impl TextSegmentor {
    /// Create a new text segmentor from the given configuration.
    ///
    /// ## Arguments
    /// * `config` - The segmentation configuration.
    ///
    /// ## Returns
    /// A new `TextSegmentor` instance.
    pub fn from_config<T>(
        config: SegmentationConfig<T>,
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

    /// Create a new text segmentor with the given regex pattern and special words.
    ///
    /// ## Arguments
    /// * `word_pattern` - The word split pattern.
    /// * `specials` - A slice of special word strings.
    /// * `max_pool` - The maximum size of the regex pool; if None, lib defaults are used.
    ///
    /// ## Returns
    /// A new `TextSegmentor` instance.
    pub fn from_patterns<P, S>(
        word_pattern: P,
        specials: &[S],
        max_pool: Option<NonZeroUsize>,
    ) -> Self
    where
        P: Into<RegexWrapperPattern>,
        S: AsRef<str>,
    {
        let span_re = word_pattern.into().compile().unwrap();

        let special_re = if specials.is_empty() {
            None
        } else {
            Some(exact_match_union_regex_pattern(specials).compile().unwrap())
        };

        Self::new(span_re, special_re, max_pool)
    }

    /// Create a new text segmentor with the given regex suppliers.
    ///
    /// ## Arguments
    /// * `word_regex` - The regex for word splitting.
    /// * `special_regex` - The optional regex for special word matching.
    /// * `max_pool` - The maximum size of the regex pool; if None, lib defaults are used.
    ///
    /// ## Returns
    /// A new `TextSegmentor` instance.
    pub fn new(
        word_regex: RegexWrapper,
        special_regex: Option<RegexWrapper>,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let word_re_pool = PoolToy::init(word_regex, max_pool);
        let special_re_pool = special_regex.map(|r| PoolToy::init(r, max_pool));

        Self {
            word_re_pool,
            special_re_pool,
        }
    }

    /// Get the span split regex.
    pub fn word_regex(&self) -> &RegexWrapper {
        self.word_re_pool.get()
    }

    /// Get the optional special split regex.
    pub fn special_regex(&self) -> Option<&RegexWrapper> {
        match &self.special_re_pool {
            None => None,
            Some(pool) => Some(pool.get()),
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
        match self.special_re_pool {
            None => None,
            Some(ref pool) => pool
                .get()
                .find_iter(text.as_ref())
                .next()
                .map(|m| m.range()),
        }
    }

    /// Split a chunk of text into [`SpanRef::Word`], appending to the `words` buffer.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    /// * `words` - The target buffer to append to.
    fn split_append_words(
        &self,
        text: &str,
        offset: usize,
        words: &mut Vec<SpanRef>,
    ) -> usize {
        let mut last = 0;
        for m in self.word_re_pool.get().find_iter(text) {
            let match_range = m.range();
            if last < match_range.start {
                words.push(SpanRef::Gap(last..match_range.start));
            }

            last = match_range.end;
            words.push(SpanRef::Word(offset_range(match_range, offset)));
        }
        last
    }

    /// Split a chunk of text into spans, appending to the `words` buffer.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    /// * `words` - The target buffer to append to.
    pub fn split_append_spans(
        &self,
        text: &str,
        words: &mut Vec<SpanRef>,
    ) {
        let mut current = text;
        let mut offset = 0;

        while let Some(range) = self.next_special_span(current) {
            let pre = &current[..range.start];
            let last = self.split_append_words(pre, offset, words);

            if last < range.start {
                words.push(SpanRef::Gap(offset_range(last..range.start, offset)));
            }

            words.push(SpanRef::Special(offset_range(range.clone(), offset)));

            current = &current[range.end..];
            offset += range.end;
        }

        if !current.is_empty() {
            let last = self.split_append_words(current, offset, words);

            if last < current.len() {
                words.push(SpanRef::Gap(offset_range(last..current.len(), offset)));
            }
        }
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
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.8);
        let mut words = Vec::with_capacity(capacity as usize);

        self.split_append_spans(text, &mut words);
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
        let mut words = Vec::new();
        self.split_append_spans(text, &mut words);
        words
            .into_iter()
            .filter(|m| !matches!(m, SpanRef::Gap(_)))
            .map(|w| &text[Range::<usize>::from(w)])
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
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;

    #[test]
    fn test_split_words() {
        type T = u32;

        let config: SegmentationConfig<T> =
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN)
                .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let segmentor = TextSegmentor::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

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

        let config: SegmentationConfig<T> = SegmentationConfig::from_pattern(r"\w+");

        let segmentor = TextSegmentor::from_config(config, Some(NonZeroUsize::new(1).unwrap()));

        let buf = "hello world!";
        assert_eq!(segmentor.remove_gaps(buf), "helloworld");
    }
}
