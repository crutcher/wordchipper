//! # Classified Span Engine
//!
//! The post-processing engine that transforms a stream of
//! [`TokenRole`]-classified logos tokens into correct `Word`/`Gap` spans.
//!
//! This is the core reusable component: define a logos enum, map each
//! variant to a [`TokenRole`], and feed the stream to
//! [`for_each_classified_span`].

use core::ops::Range;

use super::token_role::{TokenRole, contraction_split};
use crate::spanners::SpanRef;

/// Iterate classified logos tokens and emit Word/Gap spans with
/// post-processing corrections for regex compatibility:
///
/// 1. **Whitespace splitting**: the regex `\s+(?!\S)` backtracks so the last
///    whitespace byte becomes a prefix of the next word. We buffer Whitespace
///    tokens and split off the last byte when followed by certain tokens.
///
/// 2. **Prefix handling**: with 2+ whitespace chars before a token starting
///    with a non-letter, we merge the last whitespace byte + the non-letter
///    prefix into one span (matching how Punctuation's ` ?` absorbs a space
///    in the regex). With 1 whitespace char, it stays standalone.
///
/// 3. **Contraction splitting** (when `check_contraction` is true): regex
///    first-match picks Contraction `'T` over Letters `'The`, but logos
///    longest-match picks Letters. We detect the contraction prefix and
///    split the token.
///
/// # Arguments
///
/// * `iter` - iterator of `(TokenRole, Range<usize>)` pairs from logos
/// * `text` - the text being scanned (ranges index into this)
/// * `offset` - byte offset added to emitted span ranges
/// * `f` - callback; return `false` to halt early
///
/// # Returns
///
/// `(completed, consumed)` where `consumed` is the byte count of accepted
/// spans and `completed` indicates all spans were accepted.
///
/// # Example
///
/// ```
/// use wordchipper::spanners::{
///     SpanRef,
///     span_lexers::logos::{TokenRole, for_each_classified_span},
/// };
///
/// let text = "hello world";
/// let tokens = vec![
///     (
///         TokenRole::Word {
///             check_contraction: false,
///         },
///         0..5,
///     ), // "hello"
///     (TokenRole::Whitespace, 5..6), // " "
///     (
///         TokenRole::Word {
///             check_contraction: false,
///         },
///         6..11,
///     ), // "world"
/// ];
///
/// let mut spans = Vec::new();
/// for_each_classified_span(tokens.into_iter(), text, 0, &mut |span| {
///     spans.push(span);
///     true
/// });
///
/// assert_eq!(spans, vec![SpanRef::Word(0..5), SpanRef::Word(5..11),]);
/// ```
pub fn for_each_classified_span(
    iter: impl Iterator<Item = (TokenRole, Range<usize>)>,
    text: &str,
    offset: usize,
    f: &mut dyn FnMut(SpanRef) -> bool,
) -> (bool, usize) {
    let text = text.as_bytes();
    let mut last = 0;
    let mut pending_ws: Option<Range<usize>> = None;

    macro_rules! emit {
        (word $r:expr) => {
            if !f(SpanRef::Word(offset + $r.start..offset + $r.end)) {
                return (false, $r.start);
            }
        };
        (gap $r:expr) => {
            if !f(SpanRef::Gap(offset + $r.start..offset + $r.end)) {
                return (false, $r.start);
            }
        };
    }

    macro_rules! flush_ws_split {
        ($ws:expr) => {{
            let ws = $ws;
            debug_assert!(!ws.is_empty(), "flush_ws_split called with empty range");
            // Find start of the last character (may be multi-byte, e.g. NBSP).
            // Safety: text is &str bytes, so valid UTF-8; the scan always finds
            // a leading byte before reaching ws.start.
            let mut trim = ws.end - 1;
            while trim > ws.start && (text[trim] & 0xC0) == 0x80 {
                trim -= 1;
            }
            debug_assert!(
                (text[trim] & 0xC0) != 0x80,
                "no leading byte found in ws range"
            );
            if ws.start < trim {
                emit!(word(ws.start..trim));
            }
            trim
        }};
    }

    // Emit a Letters/Word span, splitting contractions if needed.
    macro_rules! emit_absorbing {
        ($start:expr, $end:expr, $check_contraction:expr) => {
            if $check_contraction {
                if let Some(split) = contraction_split(&text[$start..$end]) {
                    emit!(word($start..$start + split));
                    emit!(word($start + split..$end));
                } else {
                    emit!(word($start..$end));
                }
            } else {
                emit!(word($start..$end));
            }
        };
    }

    for (kind, span) in iter {
        let Range { start, end } = span;

        if last < start {
            if let Some(ws) = pending_ws.take() {
                emit!(word ws);
            }
            emit!(gap(last..start));
        }

        last = end;

        match kind {
            TokenRole::Whitespace => {
                if let Some(ws) = pending_ws.take() {
                    emit!(word ws);
                }
                pending_ws = Some(start..end);
            }
            TokenRole::Punctuation => {
                // Regex ` ?[^\s\p{L}\p{N}]+` absorbs a preceding ASCII
                // space (literal ` ?`). Non-space whitespace (NBSP, tab)
                // is NOT absorbed.
                if let Some(ws) = pending_ws.take() {
                    let ws_start = ws.start;
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    if trim == ws_start || text[trim] != b' ' {
                        // Single ws char, or last char is not ASCII space.
                        emit!(word(trim..ws_end));
                        emit!(word(start..end));
                    } else {
                        emit!(word(trim..end));
                    }
                } else {
                    emit!(word(start..end));
                }
            }
            TokenRole::Word { check_contraction } => {
                if let Some(ws) = pending_ws.take() {
                    let ws_start = ws.start;
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    let single_char = trim == ws_start;

                    // Decode only the first UTF-8 char from bytes to avoid
                    // validating the entire tail (which would be O(n^2) overall).
                    let first_is_letter = {
                        let tail = &text[start..];
                        let char_len = match tail.first() {
                            Some(&b) if b < 0x80 => 1,
                            Some(&b) if b < 0xE0 => 2,
                            Some(&b) if b < 0xF0 => 3,
                            Some(_) => 4,
                            None => 0,
                        };
                        char_len > 0
                            && core::str::from_utf8(&tail[..char_len])
                                .ok()
                                .and_then(|s| s.chars().next())
                                .is_some_and(char::is_alphabetic)
                    };

                    if first_is_letter {
                        // Token has no existing prefix; merge last ws char.
                        emit_absorbing!(trim, end, check_contraction);
                    } else if single_char {
                        // Single ws char: emit standalone, token as-is.
                        emit!(word(trim..ws_end));
                        emit_absorbing!(start, end, check_contraction);
                    } else {
                        // 2+ ws chars: merge last ws char + non-letter
                        // prefix into one span (like Punctuation ` ?X`),
                        // then emit remaining letters separately.
                        let prefix_len = core::str::from_utf8(&text[start..end])
                            .expect("text is &str bytes, always valid UTF-8")
                            .chars()
                            .next()
                            .map_or(1, char::len_utf8);
                        emit!(word(trim..start + prefix_len));
                        emit_absorbing!(start + prefix_len, end, check_contraction);
                    }
                } else {
                    emit_absorbing!(start, end, check_contraction);
                }
            }
            TokenRole::Standalone => {
                if let Some(ws) = pending_ws.take() {
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    emit!(word(trim..ws_end));
                }
                emit!(word(start..end));
            }
            TokenRole::Gap => {
                if let Some(ws) = pending_ws.take() {
                    emit!(word ws);
                }
                emit!(gap(start..end));
            }
        }
    }

    if let Some(ws) = pending_ws.take() {
        last = ws.end;
        emit!(word ws);
    }

    if last < text.len() {
        emit!(gap(last..text.len()));
        last = text.len();
    }

    (true, last)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::alloc::vec::Vec;

    /// Collect spans from for_each_classified_span for testing.
    fn collect_spans(
        tokens: impl Iterator<Item = (TokenRole, Range<usize>)>,
        text: &str,
        offset: usize,
    ) -> Vec<SpanRef> {
        let mut spans = Vec::new();
        for_each_classified_span(tokens, text, offset, &mut |span| {
            spans.push(span);
            true
        });
        spans
    }

    /// Assert structural invariants on a span sequence over `text`:
    /// contiguous, complete coverage, UTF-8 aligned. Allows empty spans
    /// because random role assignments can produce them (real-lexer tests
    /// in cl100k/o200k use strict non-empty checks).
    fn assert_structural_invariants(
        spans: &[SpanRef],
        text: &str,
        offset: usize,
    ) -> Result<(), TestCaseError> {
        let len = text.len();
        if spans.is_empty() {
            prop_assert_eq!(len, 0, "no spans emitted for non-empty text");
            return Ok(());
        }

        prop_assert_eq!(
            spans[0].range().start,
            offset,
            "first span doesn't start at offset"
        );
        prop_assert_eq!(
            spans.last().unwrap().range().end,
            offset + len,
            "last span doesn't end at offset + text.len()"
        );

        let bytes = text.as_bytes();
        for i in 0..spans.len() {
            let range = spans[i].range();
            prop_assert!(
                range.start <= range.end,
                "inverted span at index {}: {:?}",
                i,
                range
            );
            let local_start = range.start - offset;
            let local_end = range.end - offset;
            prop_assert!(
                core::str::from_utf8(&bytes[local_start..local_end]).is_ok(),
                "non-UTF-8 span at index {}: {:?}",
                i,
                range
            );
            if i + 1 < spans.len() {
                prop_assert_eq!(
                    range.end,
                    spans[i + 1].range().start,
                    "gap between spans {} and {}",
                    i,
                    i + 1
                );
            }
        }
        Ok(())
    }

    /// Partition `text` into char-aligned chunks using `chunk_roles`.
    /// Each entry in `chunk_roles` is `(char_count, role_index)`.
    /// Returns a vec of `(TokenRole, Range<usize>)`.
    fn build_token_stream(
        text: &str,
        chunk_roles: &[(usize, u8)],
    ) -> Vec<(TokenRole, Range<usize>)> {
        let roles = [
            TokenRole::Whitespace,
            TokenRole::Punctuation,
            TokenRole::Word {
                check_contraction: false,
            },
            TokenRole::Word {
                check_contraction: true,
            },
            TokenRole::Standalone,
            TokenRole::Gap,
        ];

        let mut tokens = Vec::new();
        let mut char_iter = text.char_indices().peekable();

        for &(char_count, role_idx) in chunk_roles {
            if char_iter.peek().is_none() {
                break;
            }
            let start = char_iter.peek().unwrap().0;
            let mut end = start;
            for _ in 0..char_count {
                if let Some((_, ch)) = char_iter.next() {
                    end += ch.len_utf8();
                } else {
                    break;
                }
            }
            if start < end {
                let role = roles[role_idx as usize % roles.len()];
                tokens.push((role, start..end));
            }
        }

        // Cover remaining text as a final token
        if let Some(&(pos, _)) = char_iter.peek() {
            let end = text.len();
            if pos < end {
                tokens.push((
                    TokenRole::Word {
                        check_contraction: false,
                    },
                    pos..end,
                ));
            }
        }

        tokens
    }

    // -------------------------------------------------------------------
    // Structural invariant proptests
    // -------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn structural_invariants_multi_role(
            text in "\\PC{0,100}",
            chunks in proptest::collection::vec((1..5usize, 0..6u8), 1..20),
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let spans = collect_spans(tokens.into_iter(), &text, 0);
            assert_structural_invariants(&spans, &text, 0)?;
        }

        #[test]
        fn structural_invariants_with_offset(
            text in "\\PC{1,50}",
            chunks in proptest::collection::vec((1..5usize, 0..6u8), 1..10),
            offset in 0..1000usize,
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let spans = collect_spans(tokens.into_iter(), &text, offset);
            assert_structural_invariants(&spans, &text, offset)?;
        }

        #[test]
        fn structural_invariants_empty_stream(text in "\\PC{0,100}") {
            let spans = collect_spans(core::iter::empty(), &text, 0);

            if text.is_empty() {
                prop_assert!(spans.is_empty());
            } else {
                prop_assert_eq!(spans.len(), 1);
                prop_assert_eq!(spans[0].clone(), SpanRef::Gap(0..text.len()));
            }
        }

        /// Early termination: stop after N spans, verify consumed is
        /// a valid byte position and accepted spans are contiguous.
        #[test]
        fn early_termination(
            text in "\\PC{1,80}",
            chunks in proptest::collection::vec((1..4usize, 0..6u8), 1..15),
            stop_after in 1..10usize,
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let mut accepted = Vec::new();
            let (completed, consumed) = for_each_classified_span(
                tokens.into_iter(),
                &text,
                0,
                &mut |span| {
                    accepted.push(span);
                    accepted.len() < stop_after
                },
            );

            if !completed {
                // Callback rejected a span: it's the last one in accepted
                prop_assert!(!accepted.is_empty());
                prop_assert!(
                    consumed <= text.len(),
                    "consumed {} > text.len() {}",
                    consumed,
                    text.len()
                );
                // Accepted spans should be contiguous from offset 0
                for i in 1..accepted.len() {
                    prop_assert_eq!(
                        accepted[i - 1].range().end,
                        accepted[i].range().start,
                        "gap between accepted spans {} and {}",
                        i - 1,
                        i
                    );
                }
            }
        }

        /// Same input always produces the same output.
        #[test]
        fn deterministic(
            text in "\\PC{0,80}",
            chunks in proptest::collection::vec((1..4usize, 0..6u8), 1..15),
        ) {
            let tokens1 = build_token_stream(&text, &chunks);
            let tokens2 = build_token_stream(&text, &chunks);
            let spans1 = collect_spans(tokens1.into_iter(), &text, 0);
            let spans2 = collect_spans(tokens2.into_iter(), &text, 0);
            prop_assert_eq!(&spans1, &spans2);
        }
    }
}
