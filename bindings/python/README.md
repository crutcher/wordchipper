# wordchipper

[![PyPI](https://img.shields.io/pypi/v/wordchipper)](https://pypi.org/project/wordchipper/)
[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)

Python bindings for the [wordchipper](https://github.com/zspacelabs/wordchipper) BPE tokenizer
library, by [ZSpaceLabs](https://zspacelabs.ai).

`wordchipper` is a high-performance Rust byte-pair encoder tokenizer for the OpenAI GPT-2 tokenizer
family. Under Python wrappers, we see a range of ~2x-4x (4 to 64 cores) speedups over
[tiktoken](https://github.com/openai/tiktoken).

| x 64 Core   | r50k python | o200k python |
|-------------|-------------|--------------|
| wordchipper | 110.5 MiB/s | 106.5 MiB/s  |
| tiktoken    | 25.5 MiB/s  | 32.7 MiB/s   |
| tokenizers  | 20.8 MiB/s  | 23.2 MiB/s   |

Read the full performance paper:
[wordchipper: Fast BPE Tokenization with Substitutable Internals](https://zspacelabs.ai/wordchipper/articles/substitutable/)

## Installation

```bash
pip install wordchipper
```

## Usage

```python
from wordchipper import Tokenizer

# See available models
Tokenizer.available_models()
# ['r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base', 'o200k_harmony']

# Load a tokenizer
tok = Tokenizer.from_pretrained("cl100k_base")

# Encode / decode
tokens = tok.encode("hello world")  # [15339, 1917]
text = tok.decode(tokens)  # "hello world"

# Batch encode / decode (parallel via rayon)
results = tok.encode_batch(["hello", "world", "foo bar"])
texts = tok.decode_batch(results)

# Vocab inspection
tok.vocab_size  # 100256
tok.token_to_id("hello")  # 15339
tok.id_to_token(15339)  # "hello"
tok.token_to_id("nonexistent")  # None

# Special tokens
tok.get_special_tokens()
# [('<|endoftext|>', 100257), ...]

# Save vocab to file (base64 tiktoken format, excludes special tokens)
tok.save_base64_vocab("vocab.tiktoken")
```

## Custom Models

Load a custom tokenizer from a tiktoken-format file with special tokens:

```python
from wordchipper import Tokenizer

# Load from tiktoken file (e.g., for Qwen, Llama, or other models)
tok = Tokenizer.from_tiktoken_file(
    path="qwen3.5.tiktoken",
    pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
    special_tokens={"<|im_start|>": 151857, "<|im_end|>": 151858},
)

tokens = tok.encode("Hello <|im_start|>system")
```

## Special Tokens Control

Filter which special tokens are included during encoding:

```python
from wordchipper import SpecialFilter

# Include all special tokens in encoding (default)
tokens = tok.encode(text, special_filter=SpecialFilter.include_all())

# Exclude all special tokens
tokens = tok.encode(text, special_filter=SpecialFilter.include_none())

# Include only specific special tokens
tokens = tok.encode(text, special_filter=SpecialFilter.include(["<|end|>"]))
```

## Drop-in compatibility

wordchipper ships drop-in replacements for both `tiktoken` and HuggingFace `tokenizers`.
Change one import line and the rest of your code stays the same.

### tiktoken

```python
# Before
import tiktoken

# After
from wordchipper.compat import tiktoken
```

Everything you use works out of the box:

```python
enc = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.encoding_for_model("gpt-4o")

tokens = enc.encode("hello world")
text = enc.decode(tokens)

# Special token handling (same defaults as tiktoken)
enc.encode("<|endoftext|>")              # raises ValueError
enc.encode("<|endoftext|>", allowed_special="all")  # [100257]
enc.encode("<|endoftext|>", disallowed_special=())  # BPE subwords

# Single-token operations
enc.encode_single_token("hello")         # 15339
enc.decode_single_token_bytes(15339)     # b'hello'

# Byte-level decoding
enc.decode_bytes(tokens)                 # b'hello world'
enc.decode_tokens_bytes(tokens)          # [b'hello', b' world']

# Batch operations (parallel via Rust)
enc.encode_batch(["hello", "world"])
enc.decode_batch([[15339], [1917]])

# Inspection
enc.is_special_token(100257)             # True
enc.token_byte_values()                  # bytes for every token in vocab
enc.n_vocab                              # 100277
enc.special_tokens_set                   # {'<|endoftext|>', ...}
```

Supported encodings: `cl100k_base`, `o200k_base`, `p50k_base`, `p50k_edit`, `r50k_base`.
Model mapping covers GPT-4o, GPT-4, GPT-3.5, o3, o1, and all legacy models.

### HuggingFace tokenizers

```python
# Before
from tokenizers import Tokenizer

# After
from wordchipper.compat.tokenizers import Tokenizer
```

```python
tok = Tokenizer.from_pretrained("Xenova/gpt-4o")

enc = tok.encode("hello world")
enc.ids                                  # [24912, 2375]
enc.tokens                               # ['hello', ' world']
enc.attention_mask                       # [1, 1]
enc.type_ids                             # [0, 0]
len(enc)                                 # 2

# Sentence pairs
enc = tok.encode("hello", pair="world")
enc.type_ids                             # [0, 1]

# Batch (supports pairs too)
tok.encode_batch(["hello", ("a", "b")])

# Decode with special token control
tok.decode(enc.ids, skip_special_tokens=True)
tok.decode(enc.ids, skip_special_tokens=False)

# Padding and truncation
tok.enable_padding(length=128, pad_id=0)
tok.enable_truncation(max_length=512)

# Vocab
tok.get_vocab()                          # {'hello': 24912, ...}
tok.get_vocab_size()                     # 200000
tok.token_to_id("hello")                # 24912
```

Mapped identifiers: `Xenova/gpt-4o`, `Xenova/gpt-4`, `Xenova/cl100k_base`, `Xenova/o200k_base`.
You can also pass bare encoding names like `cl100k_base` directly. All supported identifiers
resolve to vocabularies embedded in the binary, so `from_pretrained` never makes HTTP requests.

### Why switch?

- 2-4x faster encoding than tiktoken and HuggingFace tokenizers (see benchmarks above)
- No network requests on load (vocabs are embedded in the binary)
- Single dependency, no C compiler needed
- Both compat layers verified with side-by-side comparison tests against the upstream libraries

A few parameters are accepted for API compatibility but not yet implemented
(e.g. `is_pretokenized`). These raise `NotImplementedError` when set to non-default values.

## Development

Requires [Rust](https://rustup.rs/) and [uv](https://docs.astral.sh/uv/).

```bash
cd bindings/python

# Set up environment and build
uv venv .venv
source .venv/bin/activate
uv pip install maturin pytest
maturin develop --features python-extension-module

# Run tests
pytest tests/ -v
```

After making changes to `src/lib.rs`, rebuild with `maturin develop` before re-running tests.

## Benchmarks

Compares `wordchipper` against `tiktoken` and HuggingFace `tokenizers` for single and batch encoding
on cl100k_base and o200k_base. Uses the same corpora and methodology as the Rust benchmarks in
`wordchipper-bench`:

- **Single-string**: `english.txt` / `multilingual.txt` repeated 10x
- **Batch**: 1024 samples from fineweb-edu shard 0 (~4.2 MB)

```bash
# Install benchmark dependencies
uv pip install pytest-benchmark tiktoken tokenizers pyarrow

# Build in release mode for meaningful numbers
maturin develop --release --features python-extension-module

# Run all benchmarks
pytest benchmarks/

# Run only single-encode benchmarks
pytest benchmarks/ -k "TestSingleEncode"

# Run only batch-encode benchmarks
pytest benchmarks/ -k "TestBatchEncode"

# Run only decode benchmarks
pytest benchmarks/ -k "TestSingleDecode"

# Filter by model
pytest benchmarks/ -k "cl100k_base"
```

## License

`wordchipper` is distributed under the terms of both the MIT license and the Apache License (Version
2.0). See [LICENSE-APACHE](https://github.com/zspacelabs/wordchipper/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/zspacelabs/wordchipper/blob/main/LICENSE-MIT) for details.
