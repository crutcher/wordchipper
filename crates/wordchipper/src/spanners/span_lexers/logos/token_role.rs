//! # Token Role
//!
//! Classification of logos tokens for whitespace post-processing.
//!
//! When building a custom accelerated lexer, each logos token variant maps
//! to a [`TokenRole`] that tells the post-processing engine how the token
//! interacts with preceding whitespace.

/// How a logos token interacts with whitespace splitting.
///
/// The `OpenAI` regex patterns use `\s+(?!\S)` which backtracks so the last
/// whitespace character can be absorbed as a prefix by the next pattern
/// (e.g. `[^\r\n\p{L}\p{N}]?\p{L}+`). Logos DFA can't express lookaheads,
/// so we post-process the token stream: when a [`Whitespace`](Self::Whitespace)
/// token precedes certain token kinds, the last character merges into the
/// next span; before other tokens, it becomes a standalone word.
///
/// # Example
///
/// ```
/// use wordchipper::spanners::span_lexers::logos::TokenRole;
///
/// // Map your logos token to a role:
/// let role = TokenRole::Word {
///     check_contraction: false,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenRole {
    /// Horizontal whitespace. Buffered; last char may split to next token.
    Whitespace,
    /// Punctuation with ` ?` prefix. Always absorbs a preceding ASCII space.
    Punctuation,
    /// Letter/word token. Absorbs preceding space if token starts with a letter.
    /// When `check_contraction` is true, applies contraction-prefix splitting
    /// (e.g. `'The` -> `'T` + `he` for cl100k compatibility).
    Word {
        /// Whether to check for and split contraction prefixes.
        check_contraction: bool,
    },
    /// Token that never absorbs whitespace (digits, contractions, newlines).
    Standalone,
    /// Unrecognized bytes.
    Gap,
}

/// Check if a byte slice starts with a cl100k contraction pattern
/// (`'s`, `'t`, `'d`, `'m`, `'re`, `'ve`, `'ll`, case-insensitive)
/// followed by additional letters. Returns the split point
/// (contraction length) if the contraction is a prefix of a longer
/// word, or `None` if the input is just the contraction alone.
///
/// This is useful when building cl100k-compatible lexers where logos
/// longest-match picks `'The` as one Letters token, but the regex
/// first-match would pick Contraction `'T` then Letters `he`.
///
/// # Examples
///
/// ```
/// use wordchipper::spanners::span_lexers::logos::contraction_split;
///
/// assert_eq!(contraction_split(b"'There"), Some(2)); // split after 'T
/// assert_eq!(contraction_split(b"'llama"), Some(3)); // split after 'll
/// assert_eq!(contraction_split(b"'t"), None); // just 't, nothing after
/// assert_eq!(contraction_split(b"'re"), None); // just 're, nothing after
/// assert_eq!(contraction_split(b"hello"), None); // no apostrophe
/// ```
pub fn contraction_split(bytes: &[u8]) -> Option<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // contraction_split: exhaustive prefix testing
    // -------------------------------------------------------------------

    #[test]
    fn contraction_split_empty_and_short() {
        assert_eq!(contraction_split(b""), None);
        assert_eq!(contraction_split(b"'"), None);
        assert_eq!(contraction_split(b"'s"), None); // exactly a contraction, no trailing
    }

    #[test]
    fn contraction_split_no_apostrophe() {
        assert_eq!(contraction_split(b"hello"), None);
        assert_eq!(contraction_split(b"abc"), None);
    }

    #[test]
    fn contraction_split_single_char_suffixes() {
        // Each single-char suffix, both cases, with trailing letter
        for &suffix in &[b's', b'S', b't', b'T', b'd', b'D', b'm', b'M'] {
            let input = [b'\'', suffix, b'a'];
            assert_eq!(
                contraction_split(&input),
                Some(2),
                "expected split at 2 for {:?}",
                core::str::from_utf8(&input)
            );
        }
    }

    #[test]
    fn contraction_split_single_char_exact_length() {
        // Exactly 2 bytes after apostrophe = standalone contraction, not a split
        for &suffix in &[b's', b'S', b't', b'T', b'd', b'D', b'm', b'M'] {
            let input = [b'\'', suffix];
            assert_eq!(contraction_split(&input), None);
        }
    }

    #[test]
    fn contraction_split_two_char_suffixes() {
        let pairs: &[(u8, u8)] = &[
            (b'r', b'e'),
            (b'R', b'E'),
            (b'r', b'E'),
            (b'R', b'e'),
            (b'v', b'e'),
            (b'V', b'E'),
            (b'v', b'E'),
            (b'V', b'e'),
            (b'l', b'l'),
            (b'L', b'L'),
            (b'l', b'L'),
            (b'L', b'l'),
        ];
        for &(c1, c2) in pairs {
            let input = [b'\'', c1, c2, b'a'];
            assert_eq!(
                contraction_split(&input),
                Some(3),
                "expected split at 3 for {:?}",
                core::str::from_utf8(&input)
            );
        }
    }

    #[test]
    fn contraction_split_two_char_exact_length() {
        // Exactly 3 bytes = standalone contraction
        let pairs: &[(u8, u8)] = &[(b'r', b'e'), (b'v', b'e'), (b'l', b'l')];
        for &(c1, c2) in pairs {
            let input = [b'\'', c1, c2];
            assert_eq!(contraction_split(&input), None);
        }
    }

    #[test]
    fn contraction_split_non_contraction_prefix() {
        // Apostrophe + letter that isn't a contraction suffix
        assert_eq!(contraction_split(b"'abc"), None);
        assert_eq!(contraction_split(b"'xyz"), None);
        assert_eq!(contraction_split(b"'Hello"), None);
    }

    #[test]
    fn contraction_split_real_words() {
        assert_eq!(contraction_split(b"'There"), Some(2));
        assert_eq!(contraction_split(b"'THE"), Some(2));
        assert_eq!(contraction_split(b"'llama"), Some(3));
        assert_eq!(contraction_split(b"'velvet"), Some(3));
        assert_eq!(contraction_split(b"'really"), Some(3));
    }

    // -------------------------------------------------------------------
    // proptest: contraction_split on arbitrary bytes
    // -------------------------------------------------------------------

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]

        /// contraction_split must never panic on arbitrary byte input,
        /// and when it returns Some(n), n must be a valid split point:
        /// 0 < n < input.len().
        #[test]
        fn contraction_split_arbitrary_bytes(bytes in proptest::collection::vec(0..=255u8, 0..50)) {
            let result = contraction_split(&bytes);
            if let Some(n) = result {
                proptest::prop_assert!(n > 0, "split at 0 is invalid");
                proptest::prop_assert!(
                    n < bytes.len(),
                    "split at {} >= len {} leaves nothing after",
                    n,
                    bytes.len()
                );
            }
        }
    }
}
