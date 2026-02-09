# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/crutcher/wordchipper)

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

The current status is productionization towards an alpha release.

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

## Client Usage

### Pretrained Vocabularies

* [OpenAI OATokenizer](https://docs.rs/wordchipper/latest/wordchipper/vocab/public/openai/enum.OATokenizer.html)

### Encoders and Decoders

* [Token Encoders](https://docs.rs/wordchipper/latest/wordchipper/encoders/index.html)
* [Token Decoders](https://docs.rs/wordchipper/latest/wordchipper/decoders/index.html)

## Training Overview

* [Training Example](https://docs.rs/wordchipper/latest/wordchipper/training/index.html)

This is a code snippet overview of training.

Expect training to take ~1s/10MB of input; and to be slowed
primarily by how well the stream logic of loading the training
samples is parallelized.

Note: currently, training has limited logging, and no progress reporting.

A common training binary is probably a good idea; and much of the messiness
of supporting many different training data sources could be hidden in
the isolated deps of such a tool.

Each shard is ~90MB parquet file.

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir ~/Data/nanochat/dataset --shards 
..8 --voc
ab-size=65536 --time-encode-decode                                        
   Compiling anyhow v1.0.100
   Compiling wordchipper-disk-cache v0.2.2 (/Users/crutcher/git/wordchipper/crates/wordchipper-disk-cache)
   Compiling wordchipper-data v0.0.0 (/Users/crutcher/git/wordchipper/crates/wordchipper-data)
   Compiling wordchipper v0.2.2 (/Users/crutcher/git/wordchipper/crates/wordchipper)
   Compiling tokenizer_trainer v0.0.0 (/Users/crutcher/git/wordchipper/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 2.68s
     Running `target/release/tokenizer_trainer --dataset-dir /Users/crutcher/Data/nanochat/dataset --shards ..8 --vocab-size=65536 --time-encode-decode`
Loading Shards: [0, 1, 2, 3, 4, 5, 6, 7]
...

Training Tokenizer on shards: [0, 1, 2, 3, 4, 5, 6, 7]
- shard: 0
- shard: 1
- shard: 2
- shard: 3
- shard: 4
- shard: 5
- shard: 6
- shard: 7
- train
- training_duration: 106.70s
- vocab_size: 65535

Samples Summary:
- count: 20480
- avg size: 4741

Timing Config:
- batch size: 512

Timing Encode:
- batch avg: 18.276894ms
- sample avg: 35.697µs
- avg bps: 132.81 MB/s

Observed Bytes/Token Stats:
- total bytes: 97103222
- total tokens: 24645141
- sample byte/token: 3.94

Timing Decode:
- batch avg: 1.829894ms
- sample avg: 3.574µs
```

## Acknowledgements

* Thank you to [@karpathy](https://github.com/karpathy)
  and [nanochat](https://github.com/karpathy/nanochat)
  for the work on `rustbpe`.
* Thank you to [tiktoken](https://github.com/openai/tiktoken) for their initial work in the rust
  tokenizer space.

