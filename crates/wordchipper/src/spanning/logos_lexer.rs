//! # Logos Lexer
//!
//! A compile-time DFA-based word scanner using the `logos` crate.
//! Implements [`SpanLexer`] for cl100k and o200k patterns.

use core::ops::Range;

use logos::Logos;

use crate::spanning::{SpanLexer, SpanRef};

// ---------------------------------------------------------------------------
// Cl100k token
// ---------------------------------------------------------------------------

/// Logos token for the `cl100k_base` pattern.
///
/// | Regex branch                      | Logos variant  |
/// |-----------------------------------|----------------|
/// | `'(?i:[sdmt]\|ll\|ve\|re)`        | Contraction    |
/// | `[^\r\n\p{L}\p{N}]?\p{L}+`       | Letters        |
/// | `\p{N}{1,3}`                      | Digits         |
/// | ` ?[^\s\p{L}\p{N}]+[\r\n]*`      | Punctuation    |
/// | `\s*[\r\n]`                       | Newline        |
/// | `\s+(?!\S)` / `\s`               | Whitespace     |
#[derive(Logos, Debug, PartialEq, Clone)]
enum Cl100kToken {
    #[regex(r"'[sStTdDmM]|'[rR][eE]|'[vV][eE]|'[lL][lL]")]
    Contraction,

    #[regex(r"[^\r\n\p{Letter}\p{Number}]?\p{Letter}+")]
    Letters,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    #[regex(r" ?[^\s\p{Letter}\p{Number}]+[\r\n]*")]
    Punctuation,

    // The `+` on `[\r\n]+` is equivalent to the regex `\s*[\r\n]` in practice:
    // consecutive newlines are consumed identically because `\s*` in the regex
    // greedily eats preceding newlines. Both produce the same total span.
    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

// ---------------------------------------------------------------------------
// O200k token
// ---------------------------------------------------------------------------

// Shorthand aliases for the character classes used in o200k:
//   UPPER = [\p{Uppercase_Letter}\p{Titlecase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]
//   LOWER = [\p{Lowercase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]
//   CONTRACTION_SUFFIX = ('[sS]|'[tT]|'[dD]|'[mM]|'[rR][eE]|'[vV][eE]|'[lL][lL])?
//
// These are inlined below because logos derive macros require string literals.

/// Logos token for the `o200k_base` pattern.
///
/// Key difference from cl100k: contractions are attached to the preceding word,
/// and two word variants (uppercase-leading vs lowercase-ending) are merged
/// into a single `Word` variant because logos requires unambiguous DFA states.
///
/// | Regex branch                                         | Logos variant  |
/// |------------------------------------------------------|----------------|
/// | `[^\r\n\p{L}\p{N}]?[UPPER]*[LOWER]+CONTRACTION?`    | Word           |
/// | `[^\r\n\p{L}\p{N}]?[UPPER]+[LOWER]*CONTRACTION?`    | Word           |
/// | `\p{N}{1,3}`                                         | Digits         |
/// | ` ?[^\s\p{L}\p{N}]+[\r\n/]*`                        | Punctuation    |
/// | `\s*[\r\n]+`                                         | Newline        |
/// | `\s+`                                                | Whitespace     |
#[derive(Logos, Debug, PartialEq, Clone)]
enum O200kToken {
    // Both word patterns merged via alternation into a single regex.
    // The two overlap on \p{Modifier_Letter}/\p{Other_Letter}/\p{Mark}
    // characters (e.g. CJK), so logos requires a single DFA pattern.
    // CamelCase splitting still works: the DFA's longest-match finds
    // the same boundaries as regex first-match (e.g. "OpenAI" -> "Open"+"AI").
    // Priority > default to beat Punctuation on the [^\r\n\p{L}\p{N}]? prefix.
    #[regex(
        r"[^\r\n\p{Letter}\p{Number}]?([\p{Uppercase_Letter}\p{Titlecase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]*[\p{Lowercase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]+|[\p{Uppercase_Letter}\p{Titlecase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]+[\p{Lowercase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]*)('[sS]|'[tT]|'[dD]|'[mM]|'[rR][eE]|'[vV][eE]|'[lL][lL])?",
        priority = 3
    )]
    Word,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    // Note: o200k includes '/' in the newline-char set after punctuation.
    #[regex(r" ?[^\s\p{Letter}\p{Number}]+[\r\n/]*")]
    Punctuation,

    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

// ---------------------------------------------------------------------------
// Token classification for whitespace post-processing
// ---------------------------------------------------------------------------

/// How a logos token interacts with whitespace splitting.
///
/// The `OpenAI` regex patterns use `\s+(?!\S)` which backtracks so the last
/// whitespace character can be absorbed as a prefix by the next pattern
/// (e.g. `[^\r\n\p{L}\p{N}]?\p{L}+`). Logos DFA can't express lookaheads,
/// so we post-process the token stream: when a `Whitespace` token precedes
/// certain token kinds, the last character merges into the next span;
/// before other tokens, it becomes a standalone Word.
enum SpanKind {
    /// Horizontal whitespace (`[ \t]+`). May need splitting.
    Whitespace,
    /// Punctuation (` ?[^\s\p{L}\p{N}]+[\r\n]*`). The ` ?` prefix
    /// always absorbs a preceding space byte.
    Punctuation,
    /// cl100k Letters (`[^\r\n\p{L}\p{N}]?\p{L}+`). Absorbs a preceding
    /// space only when the token starts with a letter (no existing prefix).
    /// Also needs contraction-prefix splitting.
    Cl100kLetters,
    /// o200k Word (`[^\r\n\p{L}\p{N}]?...`). Same space-absorption rule
    /// as cl100k Letters but no contraction splitting.
    O200kWord,
    /// Token without space prefix (Digits, Contraction, Newline).
    Word,
    /// Unrecognized bytes.
    Gap,
}

impl Cl100kToken {
    fn span_kind(&self) -> SpanKind {
        match self {
            Self::Whitespace => SpanKind::Whitespace,
            Self::Letters => SpanKind::Cl100kLetters,
            Self::Punctuation => SpanKind::Punctuation,
            Self::Contraction | Self::Digits | Self::Newline => SpanKind::Word,
        }
    }
}

impl O200kToken {
    fn span_kind(&self) -> SpanKind {
        match self {
            Self::Whitespace => SpanKind::Whitespace,
            Self::Word => SpanKind::O200kWord,
            Self::Punctuation => SpanKind::Punctuation,
            Self::Digits | Self::Newline => SpanKind::Word,
        }
    }
}

/// Check if a byte slice starts with a cl100k contraction pattern
/// (`'s`, `'t`, `'d`, `'m`, `'re`, `'ve`, `'ll`, case-insensitive)
/// that extends into a longer Letters match. Returns the split point
/// (contraction length) if splitting is needed, or `None`.
fn contraction_split(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 3 || bytes[0] != b'\'' {
        return None;
    }
    let c1 = bytes[1];
    // Single-char suffixes: 's, 't, 'd, 'm
    if matches!(c1, b's' | b'S' | b't' | b'T' | b'd' | b'D' | b'm' | b'M') {
        return (bytes.len() > 2).then_some(2);
    }
    // Two-char suffixes: 're, 've, 'll.
    // Need >= 4 bytes: apostrophe + 2-char suffix + at least 1 trailing letter.
    // A 3-byte input like 're is a standalone contraction, not a split candidate.
    if bytes.len() >= 4 {
        let c2 = bytes[2];
        let is_two = matches!(
            (c1, c2),
            (b'r' | b'R', b'e' | b'E') | (b'v' | b'V', b'e' | b'E') | (b'l' | b'L', b'l' | b'L')
        );
        if is_two {
            return (bytes.len() > 3).then_some(3);
        }
    }
    None
}

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
/// 3. **Contraction splitting** (cl100k only): regex first-match picks
///    Contraction `'T` over Letters `'The`, but logos longest-match picks
///    Letters. We detect the contraction prefix and split the token.
fn for_each_logos_word(
    iter: impl Iterator<Item = (SpanKind, Range<usize>)>,
    text: &[u8],
    offset: usize,
    f: &mut dyn FnMut(SpanRef) -> bool,
) -> (bool, usize) {
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

    /// Emit a Letters/Word span, splitting contractions if needed (cl100k).
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
            SpanKind::Whitespace => {
                if let Some(ws) = pending_ws.take() {
                    emit!(word ws);
                }
                pending_ws = Some(start..end);
            }
            SpanKind::Punctuation => {
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
            SpanKind::Cl100kLetters | SpanKind::O200kWord => {
                let check = matches!(kind, SpanKind::Cl100kLetters);
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
                        emit_absorbing!(trim, end, check);
                    } else if single_char {
                        // Single ws char: emit standalone, token as-is.
                        emit!(word(trim..ws_end));
                        emit_absorbing!(start, end, check);
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
                        emit_absorbing!(start + prefix_len, end, check);
                    }
                } else {
                    emit_absorbing!(start, end, check);
                }
            }
            SpanKind::Word => {
                if let Some(ws) = pending_ws.take() {
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    emit!(word(trim..ws_end));
                }
                emit!(word(start..end));
            }
            SpanKind::Gap => {
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

// ---------------------------------------------------------------------------
// Lexer types
// ---------------------------------------------------------------------------

/// A [`SpanLexer`] for the `cl100k_base` pattern (GPT-4, GPT-3.5).
///
/// Uses a compile-time logos DFA for word scanning.
/// Special token handling is provided by [`LexerTextSpanner`](super::LexerTextSpanner)
/// composition, not by this type directly.
#[derive(Clone, Debug)]
pub struct Cl100kLexer;

/// A [`SpanLexer`] for the `o200k_base` pattern (GPT-4o).
///
/// Uses a compile-time logos DFA for word scanning.
/// Special token handling is provided by [`LexerTextSpanner`](super::LexerTextSpanner)
/// composition, not by this type directly.
#[derive(Clone, Debug)]
pub struct O200kLexer;

impl SpanLexer for Cl100kLexer {
    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        for_each_logos_word(
            Cl100kToken::lexer(text).spanned().map(|(res, range)| {
                let kind = match res {
                    Ok(tok) => tok.span_kind(),
                    Err(()) => SpanKind::Gap,
                };
                (kind, range)
            }),
            text.as_bytes(),
            offset,
            f,
        )
    }
}

impl SpanLexer for O200kLexer {
    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        for_each_logos_word(
            O200kToken::lexer(text).spanned().map(|(res, range)| {
                let kind = match res {
                    Ok(tok) => tok.span_kind(),
                    Err(()) => SpanKind::Gap,
                };
                (kind, range)
            }),
            text.as_bytes(),
            offset,
            f,
        )
    }
}

/// Look up an accelerated word lexer for a known regex pattern.
///
/// Returns `Some(lexer)` when the pattern matches a pattern for which
/// a compile-time DFA lexer exists, `None` otherwise.
pub(crate) fn lookup_word_lexer(
    pattern: &crate::regex::RegexPattern
) -> Option<crate::alloc::sync::Arc<dyn SpanLexer>> {
    use crate::{
        alloc::sync::Arc,
        pretrained::openai::{OA_CL100K_BASE_PATTERN, OA_O200K_BASE_PATTERN},
    };

    let pat = pattern.as_str();
    if pat == OA_CL100K_BASE_PATTERN.as_str() {
        Some(Arc::new(Cl100kLexer))
    } else if pat == OA_O200K_BASE_PATTERN.as_str() {
        Some(Arc::new(O200kLexer))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{string::ToString, sync::Arc, vec, vec::Vec},
        spanning::{LexerTextSpanner, TextSpanner},
    };

    /// Build a `TextSpanner` from a logos lexer with no specials.
    fn spanner(lexer: impl SpanLexer + 'static) -> LexerTextSpanner {
        LexerTextSpanner::new(Arc::new(lexer), None)
    }

    // -----------------------------------------------------------------------
    // cl100k tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_logos_basic_splitting() {
        let s = spanner(Cl100kLexer);
        let text = "hello world";
        let spans = s.split_spans(text);

        // cl100k-like: " world" is one token (space grouped with letters).
        assert_eq!(spans, vec![SpanRef::Word(0..5), SpanRef::Word(5..11),]);
    }

    #[test]
    fn test_logos_with_specials() {
        let special_pattern = crate::regex::alternate_choice_regex_pattern(&[
            "<|FNORD|>".to_string(),
            "<|NORP|>".to_string(),
        ]);
        let s = LexerTextSpanner::new(
            Arc::new(Cl100kLexer),
            Some(Arc::new(special_pattern.compile().unwrap()) as Arc<dyn SpanLexer>),
        );

        let text = "hello<|FNORD|> world<|NORP|>!";
        let spans = s.split_spans(text);

        assert_eq!(
            spans,
            vec![
                SpanRef::Word(0..5),
                SpanRef::Special(5..14),
                SpanRef::Word(14..20),
                SpanRef::Special(20..28),
                SpanRef::Word(28..29),
            ]
        );
    }

    #[test]
    fn test_logos_digits() {
        let s = spanner(Cl100kLexer);
        let text = "abc 123 4567";
        let spans = s.split_spans(text);

        assert_eq!(
            spans,
            vec![
                SpanRef::Word(0..3),
                SpanRef::Word(3..4),   // " " is Word (Whitespace token)
                SpanRef::Word(4..7),   // "123"
                SpanRef::Word(7..8),   // " " is Word (Whitespace token)
                SpanRef::Word(8..11),  // "456"
                SpanRef::Word(11..12), // "7"
            ]
        );
    }

    #[test]
    fn test_logos_contractions() {
        let s = spanner(Cl100kLexer);
        let text = "don't I'll she's";
        let spans = s.split_spans(text);

        // cl100k: "don" is letters, "'t" is contraction (separate tokens).
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();

        assert!(words.contains(&"don"));
        assert!(words.contains(&"'t"));
        assert!(words.contains(&"'ll"));
        assert!(words.contains(&"'s"));
    }

    #[test]
    fn test_logos_empty() {
        let s = spanner(Cl100kLexer);
        let spans = s.split_spans("");
        assert!(spans.is_empty());
    }

    #[test]
    fn test_logos_whitespace_only() {
        let s = spanner(Cl100kLexer);
        let text = "   ";
        let spans = s.split_spans(text);

        assert_eq!(spans, vec![SpanRef::Word(0..3)]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_logos_cl100k_unicode() {
        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanning::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let cases = [
            "Hello world",
            "Bonjour le monde",
            "Hallo Welt",
            "\u{4f60}\u{597d}\u{4e16}\u{754c}",
            "\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}",
            "price is 100 dollars",
            "caf\u{00e9} na\u{00ef}ve r\u{00e9}sum\u{00e9}",
            "Hello \u{4e16}\u{754c} 123",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            assert_eq!(
                regex_spans, logos_spans,
                "cl100k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                text, regex_spans, logos_spans
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_logos_cl100k_realworld() {
        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanning::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let cases = [
            "the Civil War\u{2014}in which",
            "nation\u{2019}s capital",
            "it\u{2019}s not easy",
            "Wade County\u{2019}s boundaries were",
            "  Like all Choctaw counties",
            "   123 numbers",
            "hello   ",
            "\t\thello",
            "  \nfoo",
            "foo  \nbar",
            "   hello world",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            if regex_spans != logos_spans {
                let regex_words: Vec<&str> = regex_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                let logos_words: Vec<&str> = logos_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                panic!(
                    "cl100k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                    text, regex_words, logos_words
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_logos_cl100k_long_text() {
        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanning::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let cases = [
            "'The items we buy are important",
            "hello\n'The quick brown fox",
            "Shakespeare's \"sources,\" then read",
            "said \"hello\" to",
            "  \"sources,\" then",
            "foo  \"bar\" baz",
            "Shakespeare's Macbeth: From Saga to Screen|\nA close reading",
            "  $400 dollars",
            "  (hello)",
            " \"sources,\" then",
            " $hello world",
            " 'The quick",
            "  'The quick",
            "  \u{2014}hello world",
            " \u{2014}hello world",
            "Shakespeare's Macbeth: From Saga to Screen|\nA close reading of Shakespeare's play that will position the play in terms of its historical and political contexts and its relation to early modern discourses on the feminine, witchcraft, and the divinity of kings. We will begin with a consideration of the historical legends that constitute Shakespeare's \"sources,\" then read the play slowly and closely, coupling our discussions with readings from the period, exploring how Shakespeare's contemporaries thought of the political and cultural issues raised in the play",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            if regex_spans != logos_spans {
                let regex_words: Vec<&str> = regex_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                let logos_words: Vec<&str> = logos_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                panic!(
                    "cl100k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                    text, regex_words, logos_words
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // o200k tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_o200k_contractions_attached() {
        let s = spanner(O200kLexer);
        let text = "don't I'll she's";
        let spans = s.split_spans(text);

        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();

        assert!(
            words.contains(&"don't"),
            "expected \"don't\" as one token, got: {:?}",
            words
        );
        assert!(
            words.contains(&" she's"),
            "expected \" she's\" as one token, got: {:?}",
            words
        );
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_o200k_unicode() {
        use crate::{
            pretrained::openai::OA_O200K_BASE_PATTERN,
            spanning::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(O200kLexer);

        let cases = [
            "Hello world",
            "Bonjour le monde",
            "\u{4f60}\u{597d}\u{4e16}\u{754c}",
            "\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}",
            "price is 100 dollars",
            "caf\u{00e9} na\u{00ef}ve r\u{00e9}sum\u{00e9}",
            "Hello \u{4e16}\u{754c} 123",
            "don't I'll she's",
            "HELLO WORLD",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            assert_eq!(
                regex_spans, logos_spans,
                "o200k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                text, regex_spans, logos_spans
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_o200k_realworld() {
        use crate::{
            pretrained::openai::OA_O200K_BASE_PATTERN,
            spanning::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(O200kLexer);

        let cases = [
            "the Civil War\u{2014}in which",
            "nation\u{2019}s capital",
            "  Like all Choctaw counties",
            "   123 numbers",
            "hello   ",
            "\t\thello",
            "  \nfoo",
            "foo  \nbar",
            "   hello world",
            "foo  \"bar\" baz",
            "$$$!!!...---",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            assert_eq!(
                regex_spans, logos_spans,
                "o200k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                text, regex_spans, logos_spans
            );
        }
    }

    // -----------------------------------------------------------------------
    // Benchmarks (require std for timing)
    // -----------------------------------------------------------------------

    #[cfg(feature = "std")]
    fn benchmark_text() -> String {
        let paragraph = "The quick brown fox jumps over the lazy dog. \
            It's a beautiful day, and I'll be taking my 3 dogs for a walk. \
            Don't forget: the temperature is 72 degrees! \
            We've been waiting since 10:30am.\n\
            \n\
            In 2024, artificial intelligence continued to advance rapidly. \
            Large language models like GPT-4 and Claude demonstrated remarkable capabilities. \
            The researchers couldn't believe the results they'd achieved.\n";
        paragraph.repeat(100)
    }

    #[cfg(feature = "std")]
    fn run_benchmark(
        label: &str,
        regex_spanner: &dyn TextSpanner,
        logos_spanner: &dyn TextSpanner,
        text: &str,
        iterations: usize,
    ) {
        use std::time::Instant;

        for _ in 0..10 {
            let _ = regex_spanner.split_spans(text);
            let _ = logos_spanner.split_spans(text);
        }

        let start = Instant::now();
        let mut regex_span_count = 0;
        for _ in 0..iterations {
            regex_span_count = regex_spanner.split_spans(text).len();
        }
        let regex_elapsed = start.elapsed();

        let start = Instant::now();
        let mut logos_span_count = 0;
        for _ in 0..iterations {
            logos_span_count = logos_spanner.split_spans(text).len();
        }
        let logos_elapsed = start.elapsed();

        let speedup = regex_elapsed.as_nanos() as f64 / logos_elapsed.as_nanos() as f64;
        let regex_mbs =
            (text.len() as f64 * iterations as f64) / regex_elapsed.as_secs_f64() / 1_000_000.0;
        let logos_mbs =
            (text.len() as f64 * iterations as f64) / logos_elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!("  [{label}]");
        eprintln!(
            "    Regex: {:?}  ({} spans, {:.1} MB/s)",
            regex_elapsed, regex_span_count, regex_mbs
        );
        eprintln!(
            "    Logos: {:?}  ({} spans, {:.1} MB/s)",
            logos_elapsed, logos_span_count, logos_mbs
        );
        eprintln!("    Speedup: {:.2}x", speedup);
    }

    /// Comparative benchmark: regex vs logos for cl100k and o200k.
    ///
    /// Run with:
    ///   cargo test -p wordchipper --release bench_regex_vs_logos -- --nocapture --ignored
    #[test]
    #[ignore]
    #[cfg(feature = "std")]
    fn bench_regex_vs_logos() {
        use crate::{
            pretrained::openai::{OA_CL100K_BASE_PATTERN, OA_O200K_BASE_PATTERN},
            spanning::{TextSpannerBuilder, TextSpanningConfig},
        };

        let text = benchmark_text();
        let iterations = 1000;

        eprintln!("\n=== Spanning Benchmark: Regex vs Logos ===");
        eprintln!(
            "Text size: {} bytes, iterations: {}\n",
            text.len(),
            iterations
        );

        let cl100k_config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let cl100k_regex = TextSpannerBuilder::new(cl100k_config).build();
        let cl100k_logos = spanner(Cl100kLexer);
        run_benchmark(
            "cl100k_base",
            cl100k_regex.as_ref(),
            &cl100k_logos,
            &text,
            iterations,
        );

        eprintln!();

        let o200k_config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN);
        let o200k_regex = TextSpannerBuilder::new(o200k_config).build();
        let o200k_logos = spanner(O200kLexer);
        run_benchmark(
            "o200k_base",
            o200k_regex.as_ref(),
            &o200k_logos,
            &text,
            iterations,
        );
    }
}
