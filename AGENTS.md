# AGENTS.md — wordchipper

This document describes the project layout and architecture of **wordchipper**. It is intended to
provide a overview context for coding agents.

## Project Overview

**wordchipper** is a high-performance Rust BPE (Byte Pair Encoding) tokenizer library targeting HPC
environments. It provides training, encoding, and decoding of BPE vocabularies, with compatibility
for `tiktoken` and `nanochat/rustbpe` formats.

- **Repository**: https://github.com/crutcher/wordchipper
- **License**: MIT
- **Rust Edition**: 2024
- **MSRV**: 1.93.0
- **Current Version**: 0.7.3

## Workspace Layout

```
wordchipper/
├── crates/
│   ├── wordchipper/           # Core library (published)
│   ├── wordchipper-disk-cache/# Download cache (published)
│   ├── wordchipper-data/      # Dataset loading (unpublished)
│   └── wordchipper-experimental/ # Experimental extensions (unpublished)
├── bindings/
│   └── python/                # Python bindings (PyO3 + maturin)
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
cargo test -p wordchipper --no-default-features --tests

# Cross-check no_std (wasm32 + ARM Cortex-M3)
cargo check -p wordchipper --target wasm32-unknown-unknown --no-default-features
cargo check -p wordchipper --target thumbv7m-none-eabi --no-default-features

# Generate docs
cargo doc --no-deps --quiet --all-features
```

CI runs all of the above on every push/PR to `main`.

### Python Bindings

```sh
cd bindings/python
uv venv .venv
source .venv/bin/activate
uv pip install maturin pytest
maturin develop
pytest tests/ -v
```

## Feature Flags

| Feature    | Purpose                                                                         |
| ---------- | ------------------------------------------------------------------------------- |
| `default`  | `foldhash` + `client` + `logos` + `rayon`                                       |
| `std`      | Standard library support (most features depend on this)                         |
| `client`   | `download` + `std`; load and run pretrained encoders/decoders                   |
| `download` | Enable vocabulary downloading via `wordchipper-disk-cache`                      |
| `training` | BPE vocabulary training (`compact_str`, `dary_heap`)                            |
| `ahash`    | Swap `HashMap`/`HashSet` to `ahash` (wins if both `ahash` + `foldhash` enabled) |
| `foldhash` | Alternative fast hash via `foldhash`                                            |
| `rayon`    | Batch parallelism wrappers for encoders/decoders                                |
| `tracing`  | `tracing` instrumentation points                                                |
| `testing`  | Test utilities for downstream crates                                            |

## `no_std` Support

The crate uses unconditional `#![no_std]` with `#[cfg(feature = "std")] extern crate std;` (see
[this PSA](https://www.reddit.com/r/rust/comments/1hs6spy/psa_for_std_feature_in_no_std_libraries/)).
A `pub(crate) mod prelude` in `lib.rs` re-exports common `alloc` types (`Vec`, `String`, `ToString`,
`Box`) so modules just add `use crate::prelude::*;` instead of individual `alloc` imports.

### Modules that work in `no_std`

The core tokenization pipeline compiles and runs without `std`:

| Module                   | What it provides                                                        |
| ------------------------ | ----------------------------------------------------------------------- |
| `spanning`               | Text splitting (regex spanners, logos DFA lexers)                       |
| `encoders`               | BPE span encoding (falls back to non-pooled mode without std)           |
| `decoders`               | Token decoding, byte/string decode results                              |
| `vocab` (core)           | `ByteMapVocab`, `SpanMapVocab`, `PairMapVocab`, `UnifiedTokenVocab`     |
| `pretrained` (constants) | Regex patterns, special token definitions                               |
| `tokenizer`              | Combined `Tokenizer` builder                                            |
| `types`                  | `TokenType` trait, `WCHashMap`/`WCHashSet` (uses `hashbrown` in no_std) |
| `errors`                 | `WCError` (without `Io` variant)                                        |
| `support` (partial)      | Regex wrappers, string/slice utilities                                  |

### Modules that require `std`

| Module                          | Feature gate | Why                                          |
| ------------------------------- | ------------ | -------------------------------------------- |
| `training/*`                    | `training`   | Depends on `compact_str`, `dary_heap`        |
| `disk_cache`                    | `download`   | File I/O, HTTP downloads                     |
| `pretrained::load_by_name`      | `download`   | Loads vocabs from disk/network               |
| `pretrained::openai::factories` | `std`        | File I/O for vocab loading                   |
| `vocab::io`                     | `std`        | Base64 vocab file reading/writing            |
| `support::concurrency`          | `std`        | `PoolToy`, thread utilities, `Mutex` pooling |
| `support::concurrency::rayon`   | `rayon`      | Batch parallelism wrappers                   |
| `support::timers`               | `std`        | Timing utilities                             |

### CI cross-compilation targets

No-std compatibility is verified in CI against real no_std targets:

```sh
# Host tests
cargo test -p wordchipper --no-default-features --tests

# Cross-compilation checks (no test runner, just type-checking)
cargo check -p wordchipper --target wasm32-unknown-unknown --no-default-features
cargo check -p wordchipper --target thumbv7m-none-eabi --no-default-features
```

## Architecture

### Core Pipeline

**Text → Spans → Tokens → Text**

1. **Spanning** (`spanning/`): Regex-based text splitting + special token handling via
   `TextSpanningConfig`. `RegexTextSpanner` implements runtime management with thread-safe regex
   pooling.
2. **Encoding** (`encoders/`): Span → token sequences via BPE merge. Use `TokenEncoderBuilder` to
   construct encoders. Multiple `SpanPolicy` implementations exist for cross-benchmarking:
   - `MergeHeapSpanPolicy` — heap-based best-merge selection
   - `MergeScanCompoundPolicy` — incremental rescan for merges
3. **Decoding** (`decoders/`): Token → bytes via dictionary lookup (`TokenDictDecoder`). Use
   `TokenDecoderBuilder` to construct decoders.
4. **Vocabulary** (`vocab/`): Layered vocab types — `ByteMapVocab`, `SpanMapVocab`, `PairMapVocab`,
   `SpecialVocab`, unified in `UnifiedTokenVocab`.

### Key Design Principles

- **Vocabularies are immutable post-construction.** This is critical for thread safety and
  cache-line performance. Dynamic span caching was tested and abandoned due to cacheline contention
  under concurrent access.
- **Multiple encoder/decoder implementations coexist** for cross-benchmarking across workloads and
  hardware.
- **Generic over `TokenType`** (`T: TokenType`). Common concrete types: `u16`, `u32`.
- **Hash strategy is swappable** via `WCHashMap`/`WCHashSet` type aliases in `types/`, controlled by
  `ahash`/ `foldhash`/`hashbrown` features.
- **Builder pattern for encoders/decoders.** Use `TokenEncoderBuilder` and `TokenDecoderBuilder` to
  construct production-ready encoder/decoder instances with appropriate parallelism configuration.

### Concurrency

- `PoolToy<T>` — thread-ID-hashed pool (avoids contention vs true thread-local storage).
- `ParallelRayonEncoder` / `ParallelRayonDecoder` — batch-level `rayon` wrappers, automatically
  included when using builders with `parallel = true` (the default).
- Thread ID hashing uses an `unsafe transmute` (documented, mirrors `tiktoken`'s approach).
- `TokenEncoder::spanner()` now returns `Arc<dyn TextSpanner>` for safe sharing of spanner instances
  across threads.

### Pretrained Models

OpenAI tokenizer support in `pretrained/openai/`:

- `OATokenizer` enum: `R50kBase`, `P50kBase`, `P50kEdit`, `Cl100kBase`, `O200kBase`, `O200kHarmony`
- Each has associated regex patterns, special tokens, and vocabulary resources.

### Python Bindings (`bindings/python/`)

PyO3 + maturin based Python package exposing the core tokenizer API:

- `src/lib.rs` — FFI layer: `Tokenizer` pyclass wrapping `UnifiedTokenVocab`,
  `Arc<dyn TokenEncoder>`, `Arc<dyn TokenDecoder>`. Errors convert via `WordchipperError` to
  `PyValueError`/`PyIOError`.
- `py_src/wordchipper/` — Python package with `__init__.py` (re-export), `__init__.pyi` (type
  stubs), `py.typed` (PEP 561 marker).
- Methods: `from_pretrained()`, `encode()`, `decode()`, `encode_batch()`, `decode_batch()`,
  `vocab_size`, `token_to_id()`, `id_to_token()`, `get_special_tokens()`, `available_models()`,
  `save_vocab()`.
- Build: `maturin develop` for dev, `maturin build --release` for wheels.

## Code Style Conventions

### "Style Hints" Pattern

Many types and type aliases carry `## Style Hints` doc comments that prescribe preferred instance
variable names. **Follow these.** Examples:

- `SpanTokenMap<T>` → instance name `span_map` or `span_token_map`
- `PoolToy<T>` → instance name `${T-name}_pool` (e.g. `regex_pool`)
- `TokenEncoder<T>` → prefer `encoder` when unambiguous
- `TokenDecoder<T>` → prefer `decoder` when unambiguous
- `TokenEncoderBuilder<T>` / `TokenDecoderBuilder<T>` → use `builder` suffixes when constructing
- `TextSpanningConfig<T>` → prefer `spanner_config` or `config`
- `RegexTextSpanner` → prefer `spanner`

### Rust Style

- **Formatter**: `rustfmt` on nightly, with custom `rustfmt.toml`:
  - `fn_params_layout = "Vertical"` — each parameter on its own line
  - `group_imports = "StdExternalCrate"` / `imports_granularity = "Crate"`
  - `format_code_in_doc_comments = true`
- **Lints**: `warnings = "deny"`, `clippy::doc_markdown = "deny"`,
  `clippy::double_must_use = "allow"`
- **`#![warn(missing_docs, unused)]`** is set at crate root.
- **Doc comments**: Use `///` with `## Arguments`, `## Returns`, `## Panics` sections as
  appropriate.
- All public items must have doc comments.
- **Constructor pattern**: Prefer `from_*` and `new()` returning `Self` or
  `wordchipper::errors::Result<Self>`.
- Builder-style methods use `with_*` naming and consume `self`.
- **Encapsulation**: Struct fields are private; provide accessor methods.
- **`cfg` gating**: Feature-gated modules use `#[cfg(feature = "...")]` at `mod` declarations.
- **`no_std` prelude**: The crate is unconditionally `#![no_std]`. Common alloc types (`Vec`,
  `String`, `ToString`, `Box`) are re-exported via `crate::prelude`. Add `use crate::prelude::*;` to
  any module that uses these types. The `vec!` and `format!` macros are available crate-wide via
  `#[macro_use] extern crate alloc` when `std` is enabled. For other alloc items, import via
  `crate::alloc::*` (e.g. `crate::alloc::sync::Arc`).

### Testing

- Tests live in `#[cfg(test)] mod tests` within the same file.
- Common test utilities are behind the `testing` feature flag.
- Use `serial_test::serial` for tests that mutate environment variables.
- Integration testing for pretrained models may require network access
  (`#[cfg(feature = "download")]`).

### Dependencies

- **Error handling**: The core `wordchipper` crate uses `thiserror` (not `anyhow`) for structured
  error types. See `crates/wordchipper/src/errors.rs` for the `WordchipperError` enum and
  `Result<T>` alias. Ancillary crates (disk-cache, examples) may still use `anyhow` at their own
  boundaries.
- **Regex pinning**: `fancy-regex = "0.13.0"` and `regex = "1.10.3"` are intentionally pinned.
  Upgrading introduces performance regressions under concurrent encoding due to contention; do not
  upgrade without benchmarking.
- `pip install` equivalent: always add deps to `[workspace.dependencies]` first, then reference with
  `{ workspace = true }` in crate-level `Cargo.toml`.

## Things to Know

- The `o200k_base` / `o200k_harmony` models use more complex regex patterns and are inherently
  slower than the `r50k`/ `p50k`/`cl100k` family.
- `tokenizers` crate benchmarks show anomalously low throughput — likely a `rayon` interaction
  issue, not yet diagnosed.
- The `no_std` path is CI-tested (host tests + wasm32/thumbv7m cross-compilation) but not yet
  validated in an actual embedded/no-std deployment.
- The project is in alpha-publication stage; API stability is not yet guaranteed.
