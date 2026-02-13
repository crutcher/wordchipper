# CLAUDE.md — wordchipper

## Project Overview

**wordchipper** is a high-performance Rust BPE (Byte Pair Encoding) tokenizer library
targeting HPC environments. It provides training, encoding, and decoding of BPE vocabularies,
with compatibility for `tiktoken` and `nanochat/rustbpe` formats.

- **Repository**: https://github.com/crutcher/wordchipper
- **License**: MIT
- **Rust Edition**: 2024
- **MSRV**: 1.93.0
- **Current Version**: 0.6.2

## Workspace Layout

```
wordchipper/
├── crates/
│   ├── wordchipper/           # Core library (published)
│   ├── wordchipper-disk-cache/# Download cache (published)
│   ├── wordchipper-data/      # Dataset loading (unpublished)
│   └── wordchipper-experimental/ # Experimental extensions (unpublished)
├── examples/
│   ├── sample-timer/          # Benchmark tool vs tiktoken/tokenizers
│   └── tokenizer_trainer/     # Training example
```

## Build & Test Commands

```sh
# Format (requires nightly)
cargo +nightly fmt --check

# Lint
cargo clippy --no-deps

# Test (full workspace)
cargo test --workspace

# Test with training feature
cargo test -p wordchipper --features training

# Test with foldhash instead of ahash
cargo test -p wordchipper --no-default-features --features client,foldhash

# Test no_std
cargo test -p wordchipper --no-default-features --features no_std --tests

# Generate docs
cargo doc --no-deps --quiet --all-features
```

CI runs all of the above on every push/PR to `main`.

## Feature Flags

| Feature    | Purpose |
|------------|---------|
| `default`  | `ahash` + `client` + `rayon` |
| `std`      | Standard library support (most features depend on this) |
| `no_std`   | `hashbrown/alloc` for no-std environments |
| `client`   | `download` + `std`; load and run pretrained encoders/decoders |
| `download` | Enable vocabulary downloading via `wordchipper-disk-cache` |
| `training` | BPE vocabulary training (`compact_str`, `dary_heap`) |
| `ahash`    | Swap `HashMap`/`HashSet` to `ahash` (wins if both `ahash` + `foldhash` enabled) |
| `foldhash` | Alternative fast hash via `foldhash` |
| `rayon`    | Batch parallelism wrappers for encoders/decoders |
| `tracing`  | `tracing` instrumentation points |
| `testing`  | Test utilities for downstream crates |

## Architecture

### Core Pipeline

**Text → Spans → Tokens → Text**

1. **Spanning** (`spanning/`): Regex-based text splitting + special token handling via `TextSpanningConfig`.
2. **Encoding** (`encoders/`): Span → token sequences via BPE merge. Multiple `SpanPolicy` implementations exist for cross-benchmarking:
   - `MergeHeapSpanPolicy` — heap-based best-merge selection
   - `MergeScanCompoundPolicy` — incremental rescan for merges
3. **Decoding** (`decoders/`): Token → bytes via dictionary lookup (`TokenDictDecoder`) or pair expansion.
4. **Vocabulary** (`vocab/`): Layered vocab types — `ByteMapVocab`, `SpanMapVocab`, `PairMapVocab`, `SpecialVocab`, unified in `UnifiedTokenVocab`.

### Key Design Principles

- **Vocabularies are immutable post-construction.** This is critical for thread safety and cache-line performance. Dynamic span caching was tested and abandoned due to cacheline contention under concurrent access.
- **Multiple encoder/decoder implementations coexist** for cross-benchmarking across workloads and hardware.
- **Generic over `TokenType`** (`T: TokenType`). Common concrete types: `u16`, `u32`.
- **Hash strategy is swappable** via `CommonHashMap`/`CommonHashSet` type aliases in `types/`, controlled by `ahash`/`foldhash`/`hashbrown` features.

### Concurrency

- `PoolToy<T>` — thread-ID-hashed pool (avoids contention vs true thread-local storage).
- `ParallelRayonEncoder` / `ParallelRayonDecoder` — batch-level `rayon` wrappers.
- Thread ID hashing uses an `unsafe transmute` (documented, mirrors `tiktoken`'s approach).

### Pretrained Models

OpenAI tokenizer support in `pretrained/openai/`:
- `OATokenizer` enum: `R50kBase`, `P50kBase`, `P50kEdit`, `Cl100kBase`, `O200kBase`, `O200kHarmony`
- Each has associated regex patterns, special tokens, and vocabulary resources.

## Code Style Conventions

### "Style Hints" Pattern

Many types and type aliases carry `## Style Hints` doc comments that prescribe preferred
instance variable names. **Follow these.** Examples:

- `SpanTokenMap<T>` → instance name `span_map` or `span_token_map`
- `PoolToy<T>` → instance name `${T-name}_pool` (e.g. `regex_pool`)
- `DefaultTokenEncoder<T>` → prefer `encoder` when unambiguous
- `DefaultTokenDecoder<T>` → prefer `decoder` when unambiguous
- `TextSpanningConfig<T>` → prefer `spanner_config` or `config`

### Rust Style

- **Formatter**: `rustfmt` on nightly, with custom `rustfmt.toml`:
  - `fn_params_layout = "Vertical"` — each parameter on its own line
  - `group_imports = "StdExternalCrate"` / `imports_granularity = "Crate"`
  - `format_code_in_doc_comments = true`
- **Lints**: `warnings = "deny"`, `clippy::doc_markdown = "deny"`, `clippy::double_must_use = "allow"`
- **`#![warn(missing_docs, unused)]`** is set at crate root.
- **Doc comments**: Use `///` with `## Arguments`, `## Returns`, `## Panics` sections as appropriate.
- All public items must have doc comments.
- **Constructor pattern**: Prefer `from_*` and `new()` returning `Self` or `anyhow::Result<Self>`.
- Builder-style methods use `with_*` naming and consume `self`.
- **Encapsulation**: Struct fields are private; provide accessor methods.
- **`cfg` gating**: Feature-gated modules use `#[cfg(feature = "...")]` at `mod` declarations.
- When using `no_std`, items from `alloc` are imported via `crate::alloc::*` (re-exported from `extern crate alloc`).

### Testing

- Tests live in `#[cfg(test)] mod tests` within the same file.
- Common test utilities are behind the `testing` feature flag.
- Use `serial_test::serial` for tests that mutate environment variables.
- Integration testing for pretrained models may require network access (`#[cfg(feature = "download")]`).

### Dependencies

- **Regex pinning**: `fancy-regex = "0.13.0"` and `regex = "1.10.3"` are intentionally pinned. Upgrading introduces performance regressions under concurrent encoding due to contention — do not upgrade without benchmarking.
- `pip install` equivalent: always add deps to `[workspace.dependencies]` first, then reference with `{ workspace = true }` in crate-level `Cargo.toml`.

## Things to Know

- The `o200k_base` / `o200k_harmony` models use more complex regex patterns and are inherently slower than the `r50k`/`p50k`/`cl100k` family.
- `tokenizers` crate benchmarks show anomalously low throughput — likely a `rayon` interaction issue, not yet diagnosed.
- The `no_std` path is CI-tested but not yet validated in an actual embedded/no-std deployment.
- The project is in alpha-publication stage; API stability is not yet guaranteed.
