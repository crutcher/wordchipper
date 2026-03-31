//! Logos DFA lexer for the Qwen3.5 pattern.
//!
//! Shared by all Qwen3.5 model sizes (0.6B–72B); they all use the same
//! tokenizer and regex pattern.
//!
//! Key differences from the OpenAI cl100k / o200k patterns:
//!
//! * **Letters + marks unified**: `[\p{L}\p{M}]+` — no UPPER/LOWER case split,
//!   marks always attach to the letter sequence rather than to punctuation.
//! * **Single-char digits**: `\p{N}` (not `\p{N}{1,3}`); each digit emits as an
//!   independent span.
//! * **Punctuation excludes marks**: `[^\s\p{L}\p{M}\p{N}]+` — combining marks
//!   are never consumed as punctuation.
//! * **Contractions are case-insensitive** and handled by the
//!   [`contraction_split`](super::gpt2_family::contraction_split) post-processor
//!   (same function used by cl100k).

use logos::Logos;

use super::gpt2_family::{
    Gpt2FamilyLogos,
    Gpt2FamilyTokenRole,
};
use crate::pretrained::openai::QWEN35_PATTERN;

/// Logos token variants for Qwen3.5.
///
/// The six variants map to the seven branches of the Qwen3.5 regex:
///
/// | Regex branch | Token | Notes |
/// |---|---|---|
/// | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)` | absorbed by `PrefixedLetters` | see below |
/// | `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` | `Letters` / `PrefixedLetters` | split on prefix presence |
/// | `\p{N}` | `Digit` | one digit at a time |
/// | ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` | `Punctuation` | |
/// | `\s*[\r\n]+` | `Newline` | |
/// | `\s+(?!\S)` + `\s+` | `Whitespace` | post-processed by span iterator |
///
/// **Contraction handling**: The Qwen3.5 contraction branch `(?i:'s|'t|…)` is
/// not a separate token. Instead, the logos DFA longest-match rule subsumes
/// short contractions (e.g. `'t`) into `PrefixedLetters`, and the
/// [`contraction_split`](super::gpt2_family::contraction_split) post-processor
/// splits cases like `'There` → `'T` + `here` exactly as the first-match regex
/// would. Standalone contractions (e.g. `'t` with no trailing letters) pass
/// through `contraction_split` unchanged (returns `None` for length-2 inputs).
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum Qwen35Token {
    // Regex branch 2a: letter/mark run without a non-letter prefix.
    // `first_char_is_letter: true` → the span iterator merges a preceding
    // trailing whitespace character into this token (simulating `[^\r\n\p{L}\p{N}]?`
    // absorbing a space).
    #[regex(r"[\p{L}\p{M}]+")]
    Letters,

    // Regex branch 2b: letter/mark run preceded by one non-letter/non-digit/
    // non-newline character (e.g. `'`, `!`, ` `).
    //
    // `check_contraction: true` triggers [`contraction_split`] when the prefix
    // is `'` and the DFA longest-match consumed more than just the contraction
    // suffix (e.g. `'There` (4 chars) beats `'T` (2 chars) by length, so the
    // DFA emits `'There` as PrefixedLetters, and the split produces `'T` + `here`).
    #[regex(r"[^\r\n\p{L}\p{N}][\p{L}\p{M}]+")]
    PrefixedLetters,

    // Regex branch 3: single Unicode digit.
    // Qwen3.5 uses `\p{N}` (one digit per match), not `\p{N}{1,3}` like cl100k.
    // Mapped to `Standalone` so it never absorbs preceding whitespace.
    #[regex(r"\p{N}")]
    Digit,

    // Regex branch 4: punctuation run with optional leading space and optional
    // trailing CR/LF. The logos DFA greedily captures `[\r\n]*` inline, so
    // the `pending_punct` trailer logic in `Gpt2FamilySpanIter` is effectively
    // inert for this pattern.
    #[regex(r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*")]
    Punctuation,

    // Regex branch 5: one or more CR/LF characters, optionally preceded by any
    // whitespace. The `\s*` greedily absorbs preceding horizontal whitespace, so
    // a `Whitespace` token never immediately precedes a `Newline` token.
    // Mapped to `Standalone` (same as o200k) — no pending-newline buffering needed.
    #[regex(r"\s*[\r\n]+")]
    Newline,

    // Horizontal whitespace only (no CR/LF). Buffered by `Gpt2FamilySpanIter`
    // to implement `\s+(?!\S)` last-char absorption semantics.
    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

impl Gpt2FamilyLogos<'_> for Qwen35Token {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            Self::Letters => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: true,
            },
            Self::PrefixedLetters => Gpt2FamilyTokenRole::Word {
                check_contraction: true,
                first_char_is_letter: false,
            },
            Self::Digit | Self::Newline => Gpt2FamilyTokenRole::Standalone,
            Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,
        }
    }
}

logos_lexer! {
    /// Logos DFA word scanner for Qwen3.5.
    pub struct Qwen35Lexer;
    token = Qwen35Token;
    pattern = QWEN35_PATTERN;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{
            sync::Arc,
            vec,
            vec::Vec,
        },
        spanners::{
            SpanRef,
            TextSpanner,
            span_lexers::{
                LexerTextSpanner,
                SpanLexer,
            },
        },
    };

    fn spanner(lexer: impl SpanLexer + 'static) -> LexerTextSpanner {
        LexerTextSpanner::new(Arc::new(lexer), None)
    }

    // ── structural invariants ────────────────────────────────────────────────

    #[test]
    fn test_qwen35_common() {
        crate::spanners::span_lexers::logos::testutil::common_lexer_tests(
            crate::alloc::boxed::Box::new(Qwen35Lexer),
        );
    }

    // ── match reference regex ────────────────────────────────────────────────

    #[cfg(feature = "testing")]
    #[test]
    fn test_qwen35_matches_reference() {
        use crate::spanners::span_lexers::accelerators::testutil::assert_matches_reference_lexer;
        use crate::support::regex::RegexPattern;

        let ref_lexer = RegexPattern::Fancy(QWEN35_PATTERN.as_str().to_string())
            .compile()
            .expect("reference pattern compiles");

        let test_lexer = Qwen35Lexer;

        let samples = &[
            "hello world",
            "  hello  world  ",
            "hello   world",
            "It's a test. Don't panic!",
            "I'm she'll they've we'd he's",
            "I'M SHE'LL THEY'VE WE'D HE'S",
            "foo123bar 456 789",
            "abc 1 2 3 def",
            "   ",
            " ",
            "",
            "a",
            "Hello, World! How are you?",
            "price is $100.00!",
            "foo   bar   baz",
            "\t\t\thello",
            "end with spaces   ",
            "\u{4e16}\u{754c}\u{4f60}\u{597d}",
            "mixed\n\n  content\there",
            "foo'bar'baz",
            "don't I'll she's",
            "'There 'The 'really",
            "'t 'T 're 'RE 'll 'll 'd 'D",
            "hello\nworld",
            "hello \n world",
            "  \n  spaces around newline  \n  ",
            "!@#$%",
            "hello!world",
            "test\r\nwindows",
            "\u{00e9}clair caf\u{00e9}",
            "e\u{0301} combining accent",
            "\u{0300}standalone mark",
        ];

        for sample in samples {
            assert_matches_reference_lexer(sample, &ref_lexer, &test_lexer);
        }
    }

    // ── behaviour tests ──────────────────────────────────────────────────────

    #[test]
    fn test_basic_splitting() {
        let s = spanner(Qwen35Lexer);
        // Space absorbed by preceding-whitespace logic, just like cl100k.
        assert_eq!(
            s.split_spans("hello world", None),
            vec![SpanRef::Word(0..5), SpanRef::Word(5..11)],
        );
    }

    #[test]
    fn test_single_digits() {
        let s = spanner(Qwen35Lexer);
        // Each digit is a separate Standalone span (Qwen uses \p{N}, not \p{N}{1,3}).
        let text = "abc123";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();
        assert_eq!(words, vec!["abc", "1", "2", "3"]);
    }

    #[test]
    fn test_digits_do_not_absorb_space() {
        let s = spanner(Qwen35Lexer);
        // Digits are Standalone: preceding space is emitted separately.
        let text = "abc 1";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();
        assert_eq!(words, vec!["abc", " ", "1"]);
    }

    #[test]
    fn test_contractions_case_insensitive() {
        let s = spanner(Qwen35Lexer);
        let text = "don't I'll SHE'S THEY'RE";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();

        // Contractions split from the preceding letters (Qwen3.5 branch 1 fires first
        // in the regex; logos uses contraction_split to replicate this).
        assert!(words.contains(&"don"), "expected \"don\" in {:?}", words);
        assert!(words.contains(&"'t"), "expected \"'t\" in {:?}", words);
        assert!(words.contains(&"'ll"), "expected \"'ll\" in {:?}", words);
        // Case-insensitive: 'S (uppercase) is still a contraction suffix.
        assert!(words.contains(&"'S"), "expected \"'S\" in {:?}", words);
        assert!(words.contains(&"'RE"), "expected \"'RE\" in {:?}", words);
    }

    #[test]
    fn test_contraction_followed_by_more_letters() {
        let s = spanner(Qwen35Lexer);
        // "'There" → logos matches "'There" (PrefixedLetters); contraction_split
        // gives "'T" + "here".
        let text = "'There";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();
        assert_eq!(words, vec!["'T", "here"]);
    }

    #[test]
    fn test_standalone_contraction() {
        let s = spanner(Qwen35Lexer);
        // "'t" alone (no trailing letters) must NOT be further split.
        assert_eq!(
            s.split_spans("'t", None),
            vec![SpanRef::Word(0..2)],
        );
        assert_eq!(
            s.split_spans("'ll", None),
            vec![SpanRef::Word(0..3)],
        );
    }

    #[test]
    fn test_marks_attach_to_letters() {
        let s = spanner(Qwen35Lexer);
        // Combining marks (U+0301 = combining acute accent) stay with the letter.
        let text = "e\u{0301}clair"; // "éclair" (e + combining acute + clair)
        let spans = s.split_spans(text, None);
        // Everything is one word (marks are in [\p{L}\p{M}]+).
        assert_eq!(spans.len(), 1);
        assert!(matches!(&spans[0], SpanRef::Word(r) if r == &(0..text.len())));
    }

    #[test]
    fn test_marks_not_punctuation() {
        let s = spanner(Qwen35Lexer);
        // A standalone combining mark (U+0300) is in [\p{L}\p{M}]+, not punctuation.
        let text = "\u{0300}";
        let spans = s.split_spans(text, None);
        assert_eq!(spans, vec![SpanRef::Word(0..text.len())]);
    }

    #[test]
    fn test_no_case_split() {
        let s = spanner(Qwen35Lexer);
        // Qwen3.5 does NOT split on case boundaries (unlike o200k).
        assert_eq!(
            s.split_spans("CamelCase", None),
            vec![SpanRef::Word(0..9)],
        );
        assert_eq!(
            s.split_spans("getElementById", None),
            vec![SpanRef::Word(0..14)],
        );
        assert_eq!(
            s.split_spans("HTMLParser", None),
            vec![SpanRef::Word(0..10)],
        );
    }

    #[test]
    fn test_newline_absorbs_preceding_whitespace() {
        let s = spanner(Qwen35Lexer);
        // "  \n" → `\s*[\r\n]+` grabs all 3 bytes as one span.
        assert_eq!(
            s.split_spans("  \n", None),
            vec![SpanRef::Word(0..3)],
        );
    }

    #[test]
    fn test_punctuation_optional_space() {
        let s = spanner(Qwen35Lexer);
        // " !" → Punctuation absorbs the leading space.
        assert_eq!(
            s.split_spans(" !", None),
            vec![SpanRef::Word(0..2)],
        );
        // "  !" → ws-split: " " emitted, then " !" absorbed by Punctuation role.
        assert_eq!(
            s.split_spans("  !", None),
            vec![SpanRef::Word(0..1), SpanRef::Word(1..3)],
        );
    }

    #[test]
    fn test_punctuation_trailing_newlines() {
        let s = spanner(Qwen35Lexer);
        // "!\n\n" → Punctuation with `[\r\n]*` captures all three bytes.
        assert_eq!(
            s.split_spans("!\n\n", None),
            vec![SpanRef::Word(0..3)],
        );
    }
}
