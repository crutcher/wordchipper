# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/crutcher/wordchipper)

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

This is ready for alpha users, and is 2x the speed of `tiktoken-rs`
for many current models.

The productionization towards an LTR stable release can be
tracked in the
[Alpha Release Tracking Issue](https://github.com/crutcher/wordchipper/issues/2).

## Encode/Decode Side-by-Side Benchmarks

| Model         | wordchipper  | tiktoken-rs  | tokenizers  |
|---------------|--------------|--------------|-------------|
| r50k_base     | 239.19 MiB/s | 169.30 MiB/s | 22.03 MiB/s |
| p50k_base     | 250.55 MiB/s | 163.07 MiB/s | 22.23 MiB/s |
| p50k_edit     | 241.69 MiB/s | 169.76 MiB/s | 21.27 MiB/s |
| cl100k_base   | 214.26 MiB/s | 125.43 MiB/s | 21.62 MiB/s |
| o200k_base    | 119.49 MiB/s | 123.75 MiB/s | 22.03 MiB/s |
| o200k_harmony | 121.80 MiB/s | 121.54 MiB/s | 22.08 MiB/s |

* *Help?* - I'm assuming some bug on my part for `tokenizers` + `rayon`.
* Methodology; 90MB shards of 1024 samples each, 48 threads.

```terminaloutput
$ for m in openai/{r50k_base,p50k_base,p50k_edit,cl100k_base,o200k_base,o200k_harmony}; \
  do RAYON_NUM_THREADS=48 cargo run --release -p sample-timer -- \
   --dataset-dir $DATASET_DIR --shards 0 --model $m; done
```

## Client Usage

### Pretrained Vocabularies

* [OpenAI OATokenizer](https://docs.rs/wordchipper/latest/wordchipper/pretrained/openai/enum.OATokenizer.html)

### Encoders and Decoders

* [Token Encoders](https://docs.rs/wordchipper/latest/wordchipper/encoders/index.html)
* [Token Decoders](https://docs.rs/wordchipper/latest/wordchipper/decoders/index.html)

## Loading Pretrained Models

Loading a pre-trained model requires reading the vocabulary,
as well as configuring the spanning (regex and special words)
configuration.

For a number of pretrained models, simplified constructors are
available to download, cache, and load the vocabulary.

See: [wordchipper::get_model](https://docs.rs/wordchipper/latest/wordchipper/get_model.html)

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    get_model,
    TokenDecoder,
    TokenEncoder,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
};

fn example() -> anyhow::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = get_model("openai/o200k_harmony", &mut disk_cache)?;

    let encoder = vocab.to_default_encoder();
    let decoder = vocab.to_default_decoder();

    Ok((encoder, decoder))
}
```

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

