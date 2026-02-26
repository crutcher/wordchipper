# no_std & Embedded

wordchipper's core tokenization pipeline works without the Rust standard library. This means you can
run BPE encoding and decoding on WASM targets, microcontrollers, and other `no_std` environments.

## What works without std

The following all work with `default-features = false`:

- Vocabulary types (`UnifiedTokenVocab`, `ByteMapVocab`, `PairMapVocab`, `SpanMapVocab`)
- Text spanning (pre-tokenization)
- BPE encoding (`SpanEncoder` implementations)
- Token decoding
- Logos DFA lexers (compile-time, no runtime regex)
- Special token handling

## What requires std

These features are behind feature flags that imply `std`:

| Feature              | Requires std because             |
| -------------------- | -------------------------------- |
| `download`           | Network I/O, file system caching |
| `datagym`            | JSON parsing, file I/O           |
| `rayon`              | Thread pool, OS threads          |
| `ahash` / `foldhash` | Standard HashMap integration     |
| Regex-based spanning | `regex` and `fancy-regex` crates |

## Configuration

```toml
[dependencies]
wordchipper = { version = "0.7", default-features = false }
```

That's it. No separate `no_std` feature flag needed. The crate uses unconditional `#![no_std]` and
conditionally links `std` when the `std` feature is active.

## How it works internally

wordchipper uses the "Reddit PSA" pattern for `no_std` support:

```rust,ignore
#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;
```

This means the crate always starts in `no_std` mode. The `std` feature adds standard library support
on top. All collection types come from `alloc` (Vec, String, Box) via an internal prelude module.

For hash maps, `hashbrown` is always available as a non-optional dependency. When `std` is active
and a fast hasher feature (`foldhash` or `ahash`) is enabled, the standard library's HashMap is used
with the fast hasher. Without `std`, hashbrown provides HashMap/HashSet.

## WASM targets

The WASM bindings (`bindings/wasm/`) demonstrate a full no_std integration. The WASM crate uses
`default-features = false` and builds the entire tokenization pipeline without `std`:

```toml
[dependencies]
wordchipper = { path = "../../crates/wordchipper", default-features = false }
```

Vocabulary loading in WASM works by parsing tiktoken-format data from `&[u8]` directly, bypassing
the `std`-gated `vocab::io` module.

CI verifies the no_std build against two targets:

```bash
# WASM (browser/Node.js)
cargo check --target wasm32-unknown-unknown --no-default-features

# ARM Cortex-M (bare metal)
cargo check --target thumbv7m-none-eabi --no-default-features
```

## Embedded considerations

On memory-constrained targets, be aware that:

- **Vocabulary size matters.** `cl100k_base` has ~100k entries. Each entry stores a byte sequence
  and merge pairs. Budget several megabytes for the vocabulary data structures.
- **No regex.** Without `std`, regex-based spanning is unavailable. The logos DFA lexers work in
  `no_std` since they're compiled at build time, but only for known patterns (r50k, cl100k, o200k).
  For custom patterns, you'd need to implement `SpanLexer` directly.
- **No parallelism.** rayon requires `std`. All encoding runs single-threaded. Use
  `SingleThreadDefault` for the best single-threaded span encoder.
- **Allocator required.** wordchipper uses `alloc` (Vec, String, Box, Arc). Your target must provide
  a global allocator.

## Example: building a vocabulary in no_std

Since `load_vocab` and file I/O require `std`, you need to construct the vocabulary from raw data.
The WASM bindings show one approach: parse a tiktoken-format `&[u8]` buffer into a `SpanMapVocab`,
combine with a `TextSpanningConfig`, and build a `UnifiedTokenVocab`.

The key types involved:

```rust,ignore
use wordchipper::{
    UnifiedTokenVocab,
    vocab::{SpanMapVocab, ByteMapVocab},
    spanners::TextSpanningConfig,
};

// 1. Parse your vocabulary data into a SpanMapVocab
// 2. Combine with a TextSpanningConfig for your pattern
// 3. Build: UnifiedTokenVocab::from_span_vocab(config, span_vocab)
```

The exact parsing depends on your data format. See `bindings/wasm/src/lib.rs` for a working
implementation.
