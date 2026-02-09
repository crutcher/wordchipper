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

## wordchipper vs tiktoken

Details:

- MacBook Pro 2023 Apple M2 Pro
- Each shard is ~90MB parquet file.
- Each encode/decode is compared for equality.

```terminaloutput
 % cargo run --release -p sample-timer -- --dataset-dir ~/datasets        
   Compiling sample-timer v0.0.0 (/Users/crutcher/git/wordchipper/examples/sample-timer)
    Finished `release` profile [optimized] target(s) in 1.16s
     Running `target/release/sample-timer --dataset-dir /Users/crutcher/datasets`
Model: "oa:o200k_harmony"
- shards: [0, 1, 2, 3]
- batch_size: 512

Samples Summary:
- num batches: 208
- avg bytes/sample: 4777
- avg bytes/token: 4.8

Encoder Times:
- wordchipper
  - batch:      31.0ms
  - sample:     60.5µs
  - bps:    75.31 MiB/s
- tiktoken
  - batch:      30.5ms
  - sample:     59.6µs
  - bps:    76.39 MiB/s

Decoder Times:
- wordchipper
  - batch:       2.0ms
  - sample:      3.9µs
  - bps:    1.14 GiB/s
- tiktoken
  - batch:       1.8ms
  - sample:      3.5µs
  - bps:    1.29 GiB/s
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

