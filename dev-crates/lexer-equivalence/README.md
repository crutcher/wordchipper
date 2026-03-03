# lexer-equivalence

Combinatorial equivalence testing for logos DFA lexers against regex reference implementations.
Validates that the accelerated lexers in `wordchipper` produce identical span boundaries to the
regex-based lexers for all three OpenAI tokenizer patterns (r50k, cl100k, o200k).

## How it works

The test generates all k-character strings (k=1..4) from a set of Unicode representative codepoints
and compares the span output of each logos lexer against the regex reference. Two characters are in
the same equivalence cell if no regex predicate in the OpenAI patterns distinguishes them, so
testing one representative per cell covers the full Unicode space.

The representative set (29 codepoints, 22 equivalence cells + 7 sub-cell extras) is defined in
`src/representatives.rs` and validated programmatically by the
`validate_representative_completeness` test.

### Test tiers

- **Strict tests** (`*_equivalence_k1_to_k4`): use a reduced representative set that excludes
  character classes with known divergences. These must always pass.
- **Full tests** (`*_full_equivalence`): use all 29 representatives. These fail until the
  corresponding logos lexer handles every divergence class.
- **Known tricky inputs**: regression tests from real-world text that exposed past divergences.

## Running

```
cargo test -p lexer-equivalence
```

The full suite tests ~732,540 inputs per lexer (29^1 + 29^2 + 29^3 + 29^4) and runs in under 10
seconds.
