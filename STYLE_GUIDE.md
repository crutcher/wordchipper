# STYLE_GUIDE.md — wordchipper

Conventions applied uniformly across all crates in the workspace, including unpublished ones.

---

## Formatting

Formatter: **nightly `rustfmt`** with the workspace `rustfmt.toml`:

- `fn_params_layout = "Vertical"` — each parameter on its own line
- `group_imports = "StdExternalCrate"` / `imports_granularity = "Crate"`
- `format_code_in_doc_comments = true`

Run: `cargo +nightly fmt`  
Check: `cargo +nightly fmt --check` (CI)

---

## Lints

The workspace `Cargo.toml` sets:

```toml
[workspace.lints.rust]
warnings = "deny"

[workspace.lints.clippy]
doc_markdown = "deny"
double_must_use = "allow"
```

`#![warn(missing_docs, unused)]` is set at each crate root. All warnings are treated as errors in
CI. Fix warnings before submitting.

---

## Documentation

All public items (`pub struct`, `pub fn`, `pub enum`, `pub trait`, constant, module) must have
`///` doc comments.

### Section Conventions

```rust
/// Short single-line summary.
///
/// Optional longer description paragraph.
///
/// ## Arguments
/// * `foo` - Description of foo.
/// * `bar` - Description of bar.
///
/// ## Returns
/// What the function returns.
///
/// ## Panics
/// When this function panics (omit if it never panics).
///
/// ## Errors
/// What errors can be returned (for functions returning `Result`).
///
/// ## Example
/// ```rust
/// // example code
/// ```
pub fn my_function(foo: u32, bar: &str) -> WCResult<String> { ... }
```

Use `ignore` on doc examples that require network access or pre-built vocabulary.

### "Style Hints"

Types that have preferred variable naming conventions carry a `## Style Hints` section in their
doc comment. **Follow these hints throughout the codebase.** Examples:

| Type | Preferred name |
|---|---|
| `SpanTokenMap<T>` | `span_map` or `span_token_map` |
| `TokenSpanMap<T>` | `token_map` or `token_span_map` |
| `TextSpanningConfig<T>` | `spanner_config` or `config` |
| `RegexTextSpanner` | `spanner` |
| `TokenEncoder<T>` | `encoder` |
| `TokenDecoder<T>` | `decoder` |
| `TokenEncoderBuilder<T>` | `encoder_builder` or `builder` |
| `PoolToy<T>` | `${T_name}_pool` (e.g. `regex_pool`) |

---

## Naming Conventions

- Types: `UpperCamelCase`
- Functions, methods, variables: `snake_case`
- Constants, statics: `SCREAMING_SNAKE_CASE`
- Feature flags: `snake_case`
- Module files mirror their public name exactly

---

## Structs and Encapsulation

Struct fields are **private**. Expose state through accessor methods:

```rust
pub struct Foo {
    bar: i32,
}

impl Foo {
    /// Get the bar value.
    pub fn bar(&self) -> i32 {
        self.bar
    }

    /// Set the bar value (in-place mutation).
    pub fn set_bar(&mut self, bar: i32) {
        self.bar = bar;
    }

    /// Set the bar value (builder-style, consumes `self`).
    pub fn with_bar(mut self, bar: i32) -> Self {
        self.set_bar(bar);
        self
    }
}
```

---

## Constructors and Builders

- Prefer `new()` returning `Self` for infallible construction.
- Prefer `from_*(...)` returning `Self` or `WCResult<Self>` for fallible or conversion
  constructors.
- Builder-style configuration methods use `with_*` naming and consume `self`.
- Dedicated builder structs use a `Builder` suffix (e.g. `TokenEncoderBuilder`).

---

## Error Handling

- **`wordchipper` crate**: use `thiserror` for structured error types. The primary error type is
  `WCError`; results are `WCResult<T>`.
- **Binaries, dev-crates, examples**: `anyhow` is acceptable.
- Do not use `unwrap()` or `expect()` in library code except in tests or with a compelling
  comment.

---

## `no_std` Discipline

The `wordchipper` crate is unconditionally `#![no_std]`:

```rust
#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;
```

- Use `use crate::prelude::*;` in any module that needs `Vec`, `String`, `ToString`, or `Box`.
- Import other `alloc` items explicitly: `use crate::alloc::sync::Arc;`
- `vec![]` and `format!()` are available crate-wide via `#[macro_use] extern crate alloc`.
- Do not introduce `std`-only constructs in `no_std`-compatible modules.
- `hashbrown` is always present (non-optional) and provides `HashMap`/`HashSet` when `std` is
  absent.

---

## Feature-Gated Code

Gate feature-dependent modules at the `mod` declaration:

```rust
#[cfg(feature = "rayon")]
pub mod parallel;
```

Conditional items within a module use `#[cfg(feature = "...")]` on the item itself.

---

## Testing

- Tests live in `#[cfg(test)] mod tests { ... }` within the same source file.
- Integration tests live in `tests/` at the crate root.
- Use `serial_test::serial` for tests that read or mutate environment variables.
- Tests behind the `testing` feature expose utilities for downstream crates.
- Tests that require network access are gated on `#[cfg(feature = "download")]`.
- Private fields may be accessed directly in tests for coverage purposes; this is intentional.

---

## Dependencies

- Library crates (`wordchipper`, `wordchipper-training`) must maintain broad semver ranges to
  avoid the "dependency matrix" problem for downstream users.
- Validate minimum supported versions with:
  ```sh
  cargo +nightly update -Z minimal-versions
  cargo test --workspace
  ```
- `regex` and `fancy-regex` are pinned to specific versions due to a performance regression under
  concurrent access. Do **not** upgrade without first benchmarking under parallel workloads.
- `hashbrown` is non-optional in `wordchipper` so that `default-features = false` works without
  extra configuration.
