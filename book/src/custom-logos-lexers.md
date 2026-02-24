# Building Custom Logos Lexers

BPE tokenizers split text in two phases: first into words (pre-tokenization), then each word into
subword tokens. The first phase is typically a big regex. OpenAI's `cl100k_base` pattern, for
example, uses alternations with Unicode property classes and lookaheads to segment "hello world"
into `["hello", " world"]`.

Regex is correct but slow. Each match backtracks through Unicode property tables, and the engine
runs single-threaded. The [logos](https://logos.maciej.codes/) crate takes a different approach: it
compiles regex patterns into a deterministic finite automaton (DFA) at build time via a derive
macro. No backtracking, no runtime regex compilation. For wordchipper's cl100k and o200k patterns,
this gives **30-50x speedups** (700+ MB/s).

But logos DFA can't express everything a regex can. The OpenAI patterns use `\s+(?!\S)`, a negative
lookahead that backtracks so the last whitespace character becomes a prefix of the next word. Logos
has no lookaheads. So we need a post-processing step that corrects the token stream after the DFA
runs.

This post-processing is extracted into composable, public building blocks. You supply the patterns.
We handle the rest.

## When to use this

The building blocks in this chapter are designed for tokenizers that use OpenAI-style regex
pre-tokenization with the `\s+(?!\S)` lookahead idiom. This includes:

- **cl100k_base** (GPT-4, GPT-3.5-turbo)
- **o200k_base** (GPT-4o)
- **p50k_base / p50k_edit** (GPT-3, Codex)
- **r50k_base** (GPT-2)
- Any custom tokenizer that copies the OpenAI regex structure

Tokenizers with fundamentally different pre-tokenization don't need this machinery:

- **SentencePiece** (Llama, Gemini) replaces spaces with `▁` before tokenization. No regex
  pre-tokenization.
- **Byte-level BPE** (GPT-NeoX, Falcon) uses HuggingFace's `ByteLevel` pre-tokenizer.
- **Bert-style** tokenizers split on whitespace and punctuation with simple rules, no lookaheads.

If your tokenizer's regex pattern doesn't use `\s+(?!\S)` or a similar whitespace-backtracking
idiom, you can still use logos for DFA speed, but you won't need `TokenRole` or
`for_each_classified_span`. Just implement `SpanLexer` directly.

## The whitespace problem, concretely

To understand why we need post-processing, consider what happens with the input `"hello   world"`.

The regex `\s+(?!\S)` matches whitespace greedily, then backtracks one character. So `"   "` (three
spaces) becomes `"  "` (two spaces as one token) + `" world"` (the last space merges into the next
word). This is how OpenAI's patterns work: whitespace "leaks" into the following word.

Logos has no lookaheads. Its DFA matches `"   "` as a single `Whitespace` token and `"world"` as a
separate `Letters` token. Without correction, you'd get different spans than the regex.

The post-processing engine fixes this. It buffers whitespace tokens and, when the next token
arrives, decides how to split them. The rules depend on what kind of token follows:

- **A letter token** (`world`): split off the last whitespace character and merge it with the word.
  Result: `"  "` + `" world"`. Matches the regex.
- **A punctuation token** (`!`): if the last whitespace character is an ASCII space, merge it with
  the punctuation (matching the ` ?` prefix in patterns like ` ?[^\s\p{L}\p{N}]+`).
- **A digit or newline**: emit the whitespace as its own span. No merging.

You don't need to implement any of this. You just tell the engine what kind of token each logos
variant represents, and it applies the correct rule.

## The building blocks

Three public items in `wordchipper::spanners::span_lexers::logos`:

### `TokenRole`

An enum that classifies how each logos token interacts with preceding whitespace:

```rust
pub enum TokenRole {
    Whitespace,
    Punctuation,
    Word { check_contraction: bool },
    Standalone,
    Gap,
}
```

Each variant tells the engine a different whitespace rule:

- **`Whitespace`**: this token _is_ whitespace. The engine buffers it and decides later how to split
  it based on what comes next.
- **`Punctuation`**: absorbs a preceding ASCII space. The engine merges the last buffered space
  character into this token's span (matching the ` ?` regex prefix).
- **`Word`**: absorbs a preceding space _if the token starts with a letter_. If the token starts
  with a non-letter prefix (like `"hello` where `"` is the first char), the engine handles the
  prefix separately. The `check_contraction` field enables cl100k-style splitting where `'The`
  becomes `'T` + `he`.
- **`Standalone`**: never absorbs whitespace. Digits, explicit contractions, newlines. Any preceding
  whitespace becomes its own span.
- **`Gap`**: unrecognized bytes. Use this for logos `Err(())`.

### `for_each_classified_span`

The engine function. You feed it a stream of `(TokenRole, Range<usize>)` pairs from your logos
lexer, and it emits corrected `SpanRef::Word` / `SpanRef::Gap` spans:

```rust
pub fn for_each_classified_span(
    iter: impl Iterator<Item = (TokenRole, Range<usize>)>,
    text: &str,
    offset: usize,
    f: &mut dyn FnMut(SpanRef) -> bool,
) -> (bool, usize)
```

The `iter` parameter is where your logos lexer plugs in. Each item is a token role paired with its
byte range in the input text. The `f` callback receives the corrected spans one at a time. The
`offset` parameter shifts all emitted ranges by a fixed amount (useful when scanning a slice of a
larger document).

The return value `(completed, consumed)` tells you whether the callback accepted all spans
(`completed`) and how many bytes were processed (`consumed`).

### `contraction_split`

An optional utility for cl100k-compatible lexers. Logos longest-match picks `'The` as one token, but
cl100k's regex first-match picks `'T` (contraction) then `he` (letters). This function detects the
contraction prefix and returns the split point.

Most custom lexers won't need this. Set `check_contraction: false` and ignore it.

## Building a custom lexer: step by step

Let's build a lexer from scratch. We'll target a simplified pattern that handles letters, digits,
punctuation, and whitespace.

### Step 1: Define the logos token enum

```rust
use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
enum MyToken {
    #[regex(r"\p{Letter}+")]
    Letters,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    #[regex(r" ?[^\s\p{Letter}\p{Number}]+")]
    Punctuation,

    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}
```

Each variant maps to a regex fragment. Logos compiles all of them into a single DFA at build time.

### Step 2: Map each variant to a `TokenRole`

This is where you make design decisions. For each token type, ask: "How should this interact with
preceding whitespace?"

```rust
use wordchipper::spanners::span_lexers::logos::TokenRole;

impl MyToken {
    fn role(&self) -> TokenRole {
        match self {
            // Whitespace is buffered; last char may merge into next token.
            Self::Whitespace => TokenRole::Whitespace,

            // Letters absorb a preceding space when the token starts with
            // a letter. No contraction splitting needed for our pattern.
            Self::Letters => TokenRole::Word { check_contraction: false },

            // Punctuation absorbs a preceding ASCII space (the ` ?` prefix).
            Self::Punctuation => TokenRole::Punctuation,

            // Digits and newlines stand alone. They never merge with
            // preceding whitespace.
            Self::Digits | Self::Newline => TokenRole::Standalone,
        }
    }
}
```

The key insight: you don't need to understand the whitespace-splitting algorithm. You just need to
know what each token _is_, and `TokenRole` maps that to the correct behavior.

### Step 3: Implement `SpanLexer`

Wire the logos lexer to `for_each_classified_span`:

```rust
use wordchipper::spanners::{SpanRef, span_lexers::SpanLexer};
use wordchipper::spanners::span_lexers::logos::{TokenRole, for_each_classified_span};

#[derive(Clone, Debug)]
pub struct MyLexer;

impl SpanLexer for MyLexer {
    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        for_each_classified_span(
            MyToken::lexer(text).spanned().map(|(res, range)| {
                let role = match res {
                    Ok(tok) => tok.role(),
                    Err(()) => TokenRole::Gap,
                };
                (role, range)
            }),
            text,
            offset,
            f,
        )
    }
}
```

That's it. ~30 lines of code for the entire lexer, and all the whitespace correction logic is
handled for you.

### The real thing: cl100k in 80 lines

The built-in `Cl100kLexer` follows exactly this pattern. The token enum has 6 variants. The `role()`
method is 7 lines. The `SpanLexer` impl is 15 lines. Everything else is the logos regex annotations
and tests. You can read the full source at
`crates/wordchipper/src/spanners/span_lexers/logos/cl100k.rs`.

## TokenRole reference

| Variant                      | Absorbs preceding whitespace?      | Use for                                                     |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| `Whitespace`                 | N/A (is whitespace)                | Horizontal whitespace tokens (`[ \t]+`, `[^\S\r\n]+`)       |
| `Punctuation`                | Yes, ASCII space only              | Punctuation with ` ?` prefix (` ?[^\s\p{L}\p{N}]+`)         |
| `Word { check_contraction }` | Yes, if token starts with a letter | Letter/word tokens (`\p{L}+`, case-sensitive word patterns) |
| `Standalone`                 | No                                 | Digits, contractions, newlines, anything that stands alone  |
| `Gap`                        | No                                 | Unrecognized bytes (logos errors). Always use for `Err(())` |

**When in doubt**, use `Standalone`. It's the safest default: the token is emitted as-is, and any
preceding whitespace becomes its own span.

## Testing your lexer

The strongest correctness guarantee is an **oracle test**: run the same input through both a
regex-based spanner and your logos lexer, and assert the spans are identical. With proptest, you can
do this over thousands of random Unicode strings:

```rust
use proptest::prelude::*;

#[test]
#[cfg(feature = "std")]
fn my_lexer_matches_regex() {
    // Build a regex spanner from the same pattern your logos enum targets.
    let regex_spanner = /* ... */;
    let logos_spanner = /* LexerTextSpanner wrapping MyLexer */;

    let config = proptest::test_runner::Config::with_cases(2000);
    proptest!(config, |(text in "\\PC{0,200}")| {
        let regex_spans = regex_spanner.split_spans(&text);
        let logos_spans = logos_spanner.split_spans(&text);
        prop_assert_eq!(&regex_spans, &logos_spans);
    });
}
```

If 2000 random Unicode inputs produce identical output, you have high confidence the lexer is
correct. The built-in cl100k and o200k lexers both pass this test.
