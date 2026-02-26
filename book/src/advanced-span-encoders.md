# Advanced: Span Encoders

This chapter covers wordchipper's span encoder architecture: what a span encoder does, how the five
built-in implementations work, and when you might choose one over another.

## What a span encoder does

After pre-tokenization splits text into spans, each span needs to be converted into a sequence of
token IDs. This is the job of the `SpanEncoder` trait:

```rust,ignore
pub trait SpanEncoder<T: TokenType>: Send {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    );
}
```

The encoder receives a byte slice (one span) and appends the resulting token IDs to a buffer. The
vocabulary provides two resources for this:

1. **Span lookup** (`lookup_token`): Check if the entire span is a single known token. If so, return
   it directly. This is the fast path.
2. **Pair merge table** (`pair_vocab`): A map from `(token_a, token_b) -> merged_token`. This drives
   the BPE merge loop.

When a span is a single token, encoding is a single hash map lookup. When it's not, the encoder must
run BPE: start with byte-level tokens and iteratively merge pairs according to the merge table's
rank ordering.

## The BPE merge problem

The core BPE algorithm is simple to state: repeatedly merge the lowest-ranked pair in the sequence.
But implementing it efficiently is tricky.

Consider the span `"abcd"` with merges ranked: `(a,b)=0, (c,d)=1, (ab,cd)=2`.

```text
Start:   [a, b, c, d]
Step 1:  [ab, c, d]       merge (a,b) at rank 0
Step 2:  [ab, cd]          merge (c,d) at rank 1
Step 3:  [abcd]            merge (ab,cd) at rank 2
```

The challenge is that after each merge, the neighboring pairs change. Merging `(a,b)` into `ab`
creates a new pair `(ab,c)` that didn't exist before. A naive implementation scans the entire
sequence after each merge to find the next lowest-ranked pair, giving O(n^2) behavior.

### Why shift-reduce doesn't work

A tempting optimization is a shift-reduce parser: push tokens onto a stack, and whenever the top two
tokens form a mergeable pair with a lower rank than the next input pair, reduce (merge) them. This
runs in O(n) and works for most inputs.

But it fails on some inputs because BPE requires _global_ rank ordering, not just local comparison.
A local decision to merge might miss a lower-ranked pair further to the right. This was verified
empirically against cl100k and o200k corpora.

## The five implementations

### BufferSweep (Reference)

The simplest correct implementation. Maintains a buffer of tokens and repeatedly sweeps through it,
merging the globally lowest-ranked pair in each pass.

- **Time:** O(n \* m) where m is the number of merge rounds
- **Space:** O(n)
- **Use case:** Testing and correctness comparison. Not optimized.

Selected via `SpanEncoderSelector::Reference` or `SpanEncoderSelector::BufferSweep`.

### PriorityMerge (SingleThreadDefault)

Uses a priority queue (min-heap) to track mergeable pairs. Instead of scanning the whole buffer each
round, it pops the lowest-ranked pair from the heap. After merging, it updates the neighboring pairs
in the heap.

- **Time:** O(n log n) amortized
- **Space:** O(n) for the heap
- **Use case:** Single-threaded workloads. Best single-thread performance.

Selected via `SpanEncoderSelector::SingleThreadDefault` or `SpanEncoderSelector::PriorityMerge`.

### MergeHeap (ConcurrentDefault)

Similar to PriorityMerge but optimized for concurrent access patterns. The internal state is
designed to play well when multiple threads encode different spans through the same tokenizer.

- **Time:** O(n log n) amortized
- **Space:** O(n)
- **Use case:** Multi-threaded workloads with rayon. Default for `parallel(true)`.

Selected via `SpanEncoderSelector::ConcurrentDefault` or `SpanEncoderSelector::MergeHeap`.

### TailSweep

An alternative scanning strategy that sweeps from the tail of the buffer. Useful in
memory-constrained scenarios.

- **Time:** O(n \* m)
- **Space:** O(n)
- **Use case:** When memory allocation patterns matter.

Selected via `SpanEncoderSelector::TailSweep`.

### BpeBacktrack

A fundamentally different approach. Instead of starting with bytes and merging up, it starts with
the longest possible tokens and backtracks when merge boundaries don't align.

The algorithm:

1. **Build an Aho-Corasick automaton** from all vocabulary token byte sequences.
2. **Scan the input** with leftmost-longest matching to find the longest token at each position.
3. **Validate boundaries**: check that each token pair boundary is a valid BPE merge point using
   `is_valid_token_pair`. If not, backtrack to a shorter token.
4. **Walk the next-prefix chain** when backtracking: each token has a precomputed "next longest
   prefix" that allows O(1) fallback.

This gives O(n) encoding time (amortized, since backtracking visits each byte at most twice).

- **Time:** O(n) amortized
- **Space:** O(V) for the AC automaton (built once, shared via Arc)
- **Use case:** When exact BPE semantics are required, or for very long spans.
- **Trade-off:** Higher upfront cost to build the automaton from the vocabulary.

Selected via `SpanEncoderSelector::BpeBacktrack`.

The implementation is based on the work by Hendrik van Antwerpen and Alexander Neubeck at GitHub
Next (2023), described in the blog post "So many tokens, so little time."

## Precomputed data structures

`BpeBacktrack` is the only encoder that pre-builds data structures from the vocabulary. The
`BpeVocab` struct contains:

- **`pair_lookup`**: The standard `(T, T) -> T` merge table.
- **`split_table`**: Inverse of `pair_lookup`. For each token, stores its canonical split into two
  sub-tokens.
- **`next_prefix`**: For each token, the longest prefix token that is shorter by one or more bytes.
  This enables O(1) backtracking.
- **`ac`**: The Aho-Corasick automaton over all token byte sequences.
- **`token_lens`**: Byte length of each token, indexed by token ID.

This is built once by `BpeVocab::from_vocab` and shared via `Arc` across all encoder instances.

## Choosing the right encoder

| Scenario                             | Recommended                          | Why                                        |
| ------------------------------------ | ------------------------------------ | ------------------------------------------ |
| Web server, many concurrent requests | `ConcurrentDefault`                  | MergeHeap plays well with thread pools     |
| CLI tool, single-threaded            | `SingleThreadDefault`                | PriorityMerge is fastest single-threaded   |
| Correctness testing                  | `Reference`                          | BufferSweep is simplest to reason about    |
| Very long spans (>1KB)               | `BpeBacktrack`                       | O(n) vs O(n log n) matters for long inputs |
| Memory-constrained embedded          | `TailSweep` or `SingleThreadDefault` | Lower allocation pressure                  |

For most users, the defaults (`ConcurrentDefault` when parallel, `SingleThreadDefault` otherwise)
are the right choice. The `SpanEncoderSelector` enum makes switching a one-line change for
benchmarking.

## The fast path: single-token spans

Before any BPE algorithm runs, the encoder checks whether the entire span is already a single token
in the vocabulary. This is a single hash map lookup and is O(1). For common words like "the", "is",
" world", this fast path handles the majority of spans.

The `encode_append_span_ref` method in the `SpanEncoder` trait implements this:

```rust,ignore
SpanRef::Word(range) => {
    let span = &text[range].as_bytes();
    if let Some(token) = vocab.lookup_token(span) {
        tokens.push(token);  // fast path: single lookup
    } else {
        self.encode_append_compound_span(vocab, span, tokens);  // BPE merge loop
    }
}
```

In practice, with large vocabularies like `o200k_base` (200k tokens), the vast majority of spans hit
the fast path. BPE only runs on uncommon words, misspellings, and multilingual text that doesn't
appear verbatim in the vocabulary.
