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
% RAYON_NUM_THREADS=48 cargo run --release -p sample-timer  -- \
    --dataset-dir $DATASET_CACHE_DIR --decode
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
    decode: false,
    validate: true,
    respan_input_for_decode_check: true,
}
Model: "openai/o200k_harmony"

Samples Summary:
- num batches: 104
- avg bytes/sample: 4777
- avg bytes/token: 4.8

Encoder Batch Timing:
- "wordchipper"
  - batch:      36.2ms
  - sample:     35.3µs
  - bps:    128.96 MiB/s
- "tiktoken-rs"
  - batch:      36.5ms
  - sample:     35.6µs
  - bps:    127.86 MiB/s
- "tokenizers"
  - batch:     214.7ms
  - sample:    209.6µs
  - bps:    21.73 MiB/s
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

