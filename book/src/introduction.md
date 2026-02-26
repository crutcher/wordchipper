# Introduction

## What is a tokenizer?

Large language models don't read text the way humans do. Before a model can process the sentence
"The cat sat on the mat," it must convert that string of characters into a sequence of numbers
called **tokens**. Each token is an integer that maps to an entry in a fixed **vocabulary**. The
process of converting text to tokens is called **tokenization**, and the software that does it is a
**tokenizer**.

Tokenization sits at the boundary between human-readable text and machine-readable numbers. Every
time you send a prompt to an LLM, a tokenizer runs first. Every time the model generates a response,
a tokenizer runs in reverse to turn the output tokens back into text.

### A simple example

Consider the string `"hello world"`. A tokenizer with a vocabulary like OpenAI's `cl100k_base` would
encode it as two tokens:

```text
"hello world" -> [15339, 1917]
```

The token `15339` maps to the bytes for `"hello"`, and `1917` maps to `" world"` (note the leading
space). Decoding reverses this: `[15339, 1917]` becomes the original string.

Not all strings split at word boundaries. The tokenizer might split `"tokenization"` into
`["token", "ization"]`, or keep `"the"` as a single token. The exact splits depend on the vocabulary
and algorithm.

### Why does tokenization matter?

Tokenization affects LLM behavior in ways that are easy to overlook:

- **Cost.** API pricing is per-token. A tokenizer that produces fewer tokens for the same text saves
  money. A tokenizer that splits your language inefficiently costs more.
- **Context window.** Models have a fixed token limit. Efficient tokenization means more text fits
  in the window.
- **Latency.** Each token requires a forward pass through the model. Fewer tokens means faster
  inference.
- **Multilingual fairness.** Some tokenizers handle English efficiently but split Chinese, Arabic,
  or Hindi into many more tokens for the same semantic content.
- **Reproducibility.** If you're building retrieval, caching, or evaluation pipelines, you need a
  tokenizer that produces _exactly_ the same output as the model's own tokenizer. Approximate
  doesn't cut it.

### What is BPE?

**Byte Pair Encoding (BPE)** is the most widely used tokenization algorithm for large language
models. It works by starting with individual bytes and iteratively merging the most frequent
adjacent pairs into new tokens. The result is a vocabulary of subword units that balances two
goals: common words like `"the"` become a single token, while rare words like `"defenestration"`
decompose into reusable pieces (`["def", "en", "est", "ration"]`).

BPE is used by GPT-2 through GPT-4o and most of their derivatives. The
[How Tokenizers Work](./how-tokenizers-work.md) chapter walks through the algorithm step by step.

## What is wordchipper?

**wordchipper** is a high-performance BPE tokenizer library written in Rust. It encodes and decodes
text using the same vocabularies as OpenAI's models (GPT-2, GPT-3.5, GPT-4, GPT-4o), producing
identical output to the reference implementations.

Key properties:

- **Fast.** Up to 2x the throughput of `tiktoken-rs` on encode/decode benchmarks, with 30-50x faster
  pre-tokenization via compile-time DFA lexers.
- **Correct.** Validated against OpenAI's `tiktoken` and HuggingFace `tokenizers` on large corpora.
  Token-for-token identical output.
- **Portable.** Core tokenization works in `no_std` environments. Runs on WASM, embedded targets,
  and bare-metal Rust.
- **Multi-language.** Native Rust API, plus Python and JavaScript/TypeScript bindings.
- **Extensible.** Bring your own vocabulary, write custom lexers, swap BPE algorithms.

### Supported models

| Model           | Used by          | Vocab size |
| --------------- | ---------------- | ---------- |
| `r50k_base`     | GPT-2            | ~50k       |
| `p50k_base`     | Codex            | ~50k       |
| `p50k_edit`     | Codex (edit)     | ~50k       |
| `cl100k_base`   | GPT-3.5, GPT-4   | ~100k      |
| `o200k_base`    | GPT-4o           | ~200k      |
| `o200k_harmony` | GPT-4o (harmony) | ~200k      |

### Who is this book for?

This book is for anyone who wants to understand how tokenizers work and how to use wordchipper
effectively. It's structured in layers:

- **If you're new to tokenizers**, start with [How Tokenizers Work](./how-tokenizers-work.md) for a
  visual, step-by-step explanation of BPE.
- **If you just want to encode text**, skip to [Getting Started](./getting-started.md) for a working
  example in five lines.
- **If you're optimizing throughput**, read [Performance](./performance.md) to understand the
  available BPE algorithms and DFA acceleration.
- **If you're building something custom**, see
  [Building Custom Logos Lexers](./custom-logos-lexers.md) and
  [Advanced: Span Encoders](./advanced-span-encoders.md).
