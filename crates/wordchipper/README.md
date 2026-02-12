# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/crutcher/wordchipper)

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

The current status is productionization towards an alpha release.

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

See: [wordchipper::pretrained::openai::OATokenizer](https://docs.rs/wordchipper/latest/wordchipper/pretrained/openai/enum.OATokenizer.html)

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    decoders::{DefaultTokenDecoder, TokenDecoder},
    disk_cache::WordchipperDiskCache,
    encoders::{DefaultTokenEncoder, TokenEncoder},
    pretrained::openai::OATokenizer,
    vocab::UnifiedTokenVocab,
};

fn example() -> anyhow::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
    let model = OATokenizer::O200kHarmony;
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = model.load(&mut disk_cache)?;

    let encoder: Arc<DefaultTokenEncoder<u32>> =
        DefaultTokenEncoder::new(vocab.clone(), None).into();
    let decoder: Arc<DefaultTokenDecoder<u32>> =
        DefaultTokenDecoder::from_unified_vocab(vocab).into();
        
    #[cfg(feature = "rayon")]
    use wordchipper::concurrency::rayon::*;
    
    #[cfg(feature = "rayon")]
    let encoder = Arc::new(ParallelRayonEncoder::new(encoder));

    #[cfg(feature = "rayon")]
    let decoder = Arc::new(ParallelRayonDecoder::new(decoder));

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

