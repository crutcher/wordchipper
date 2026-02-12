# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml)
[![license](https://shields.io/badge/license-MIT-blue)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/crutcher/wordchipper)

I am usually available as `@crutcher` on the Burn Discord:

* Burn
  Discord: [![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

The primary documentation is for the [wordchipper crate](crates/wordchipper).

## Encode/Decode Side-by-Side Benchmarks

```terminaloutput
$ RAYON_NUM_THREADS=48 cargo run --release -p sample-timer -- \
    --dataset-dir $DATASET_DIR --decode
Args {
    dataset_dir: "/media/Data/nanochat/dataset",
    shards: [
        0,
        1,
    ],
    batch_size: 1024,
    model: OpenaiO200kHarmony,
    ignore_missing: true,
    tiktoken: true,
    tokenizers: true,
    decode: true,
    validate: true,
    respan_input_for_decode_check: false,
}
Loaded:
- "wordchipper::openai/o200k_harmony"
- "tiktoken-rs::o200k_harmony"
- "tokenizers::Xenova/gpt-4o"

Samples Summary:
- num batches: 104
- avg bytes/sample: 4777
- avg bytes/token: 4.8

Encoder Batch Timing:
- "wordchipper::openai/o200k_harmony"
  - batch:      37.1ms
  - sample:     36.2µs
  - bps:    125.85 MiB/s
- "tiktoken-rs::o200k_harmony"
  - batch:      37.4ms
  - sample:     36.5µs
  - bps:    124.68 MiB/s
- "tokenizers::Xenova/gpt-4o"
  - batch:     201.1ms
  - sample:    196.4µs
  - bps:    23.20 MiB/s
  
Decoder Batch Timing:
- "wordchipper::openai/o200k_harmony"
  - batch:       2.9ms
  - sample:      2.9µs
  - bps:    1.55 GiB/s
- "tiktoken-rs::o200k_harmony"
  - batch:       2.2ms
  - sample:      2.1µs
  - bps:    2.12 GiB/s
- "tokenizers::Xenova/gpt-4o"
  - batch:       9.0ms
  - sample:      8.8µs
  - bps:    518.15 MiB/s
```

## Components

### Published Crates

- [wordchipper](crates/wordchipper)
- [wordchipper-disk-cache](crates/wordchipper-disk-cache)

### Unpublished Crates

- [wordchipper-data](crates/wordchipper-data)

### Tools

- [sample-timer](examples/sample-timer) - wordchipper vs tiktoken side-by-side timer.
- [tokenizer_trainer](examples/tokenizer_trainer) - training example.

## Acknowledgements

* Thank you to [@karpathy](https://github.com/karpathy)
  and [nanochat](https://github.com/karpathy/nanochat)
  for the work on `rustbpe`.
* Thank you to [tiktoken](https://github.com/openai/tiktoken) for their initial work in the rust
  tokenizer space.

