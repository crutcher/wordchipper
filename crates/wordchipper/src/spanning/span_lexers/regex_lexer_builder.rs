//! Regex-based `SpanLexer`

use core::num::NonZeroUsize;

use crate::{
    alloc::sync::Arc,
    spanning::span_lexers::SpanLexer,
    support::regex::{RegexPattern, RegexWrapper},
};

impl SpanLexer for RegexWrapper {
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)> {
        self.find_iter(&text[offset..])
            .next()
            .map(|m| (offset + m.start(), offset + m.end()))
    }
}

/// Build a regex-based [`SpanLexer`] with the given configuration.
///
/// ## Arguments
/// * `pattern` - the pattern.
/// * `concurrent` - whether to use a concurrent pool.
/// * `max_pool` - the max size of the concurrent pool;
///   `None` will use system/environment defaults.
pub fn build_regex_lexer(
    pattern: RegexPattern,
    concurrent: bool,
    max_pool: Option<NonZeroUsize>,
) -> Arc<dyn SpanLexer> {
    let _ = concurrent;
    let _ = max_pool;

    #[cfg(feature = "logos")]
    {
        use crate::spanning::span_lexers::logos::lookup_word_lexer;
        if let Some(lexer) = lookup_word_lexer(&pattern) {
            return lexer;
        }
    }

    let re: RegexWrapper = pattern.into();

    #[cfg(feature = "std")]
    if concurrent {
        return Arc::new(crate::support::concurrency::PoolToy::new(re, max_pool));
    }

    Arc::new(re)
}
