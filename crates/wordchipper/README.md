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

## Example Usage

```rust,no_run
use wordchipper::decoders::{TokenDictDecoder, TokenDecoder};
use wordchipper::encoders::{DefaultTokenEncoder, TokenEncoder};
use wordchipper::concurrency::rayon::{ParallelRayonDecoder, ParallelRayonEncoder};
use wordchipper::regex::{regex_pool_supplier, RegexWrapperPattern};
use wordchipper::spanning::{TextSpanningConfig, TextSpanner};
use wordchipper::pretrained::openai::OATokenizer;
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::disk_cache::WordchipperDiskCache;

type T = u32;

let mut disk_cache = WordchipperDiskCache::default();
let vocab: UnifiedTokenVocab<T> = OATokenizer::0200kHarmony::load(&mut disk_cache).unwrap();

let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::init(vocab.clone(), None);
let encoder = ParallelRayonEncoder::new(encoder);

let decoder = TokenDictDecoder::from_unified_vocab(vocab.clone());
let decoder = ParallelRayonDecoder::new(decoder);
```

### TokenEncoder Clients

Encoder clients should use:

* `DefaultTokenEncoder` - the current default (only?) `TokenEncoder`.
* `ParallelRayonEncoder` - a batch parallelism wrapper around any `TokenEncoder`.

```rust,no_run
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::encoders::DefaultTokenEncoder;
use wordchipper::encoders::TokenEncoder;
use wordchipper::types::TokenType;

fn example<T: TokenType>(
    vocab: &UnifiedTokenVocab<T>,
    batch: &[&str],
) -> Vec<Vec<T>> {
    let encoder = DefaultTokenEncoder::<T>::init(vocab.clone(), None);

    #[cfg(feature = "rayon")]
    let encoder = wordchipper::concurrency::rayon::ParallelRayonEncoder::new(encoder);

    encoder.try_encode_batch(batch).unwrap()
}
```

### TokenDecoder Clients

Decoder clients should use:

* `TokenDictDecoder` - the fastest `TokenDecoder`.
* `ParallelRayonDecoder` - a batch parallelism wrapper around any `TokenDecoder`.

```rust,no_run
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::decoders::DefaultTokenDecoder;
use wordchipper::decoders::TokenDecoder;
use wordchipper::types::TokenType;

fn example<T: TokenType>(
    vocab: &UnifiedTokenVocab<T>,
    batch: &[Vec<T>],
) -> Vec<String> {
    let decoder = DefaultTokenDecoder::<T>::from_unified_vocab(vocab);

    #[cfg(feature = "rayon")]
    let decoder = wordchipper::concurrency::rayon::ParallelRayonDecoder::new(decoder);

    decoder.try_decode_batch_to_strings(batch).unwrap().unwrap()
}
```

## Training Overview

See `examples/tokenizer_trainer`.

This is a code snippet overview of training.

Expect training to take ~1s/10MB of input; and to be slowed
primarily by how well the stream logic of loading the training
samples is parallelized.

Note: currently, training has limited logging, and no progress reporting.

A common training binary is probably a good idea; and much of the messiness
of supporting many different training data sources could be hidden in
the isolated deps of such a tool.

Here:

- The iterator stream for samples may be quite large.
- Training a `nanochat` equivalent tokenizer takes ~80 CPU minutes.

```rust,no_run
use wordchipper::training::bpe_trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
use wordchipper::vocab::io::tiktoken_io::save_span_map_to_tiktoken_path;
use wordchipper::pretrained::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
use wordchipper::vocab::{ByteMapVocab, UnifiedTokenVocab};
use wordchipper::encoders::DefaultTokenEncoder;
use wordchipper::decoders::DefaultTokenDecoder;
use wordchipper::concurrency::rayon::{ParallelRayonEncoder, ParallelRayonDecoder};
use std::sync::Arc;

fn example<I, S>(
    vocab_size: usize,
    batches: I,
    tiktoken_save_path: Option<String>,
) where
    I: IntoIterator,
    I::Item: AsRef<[S]>,
    S: AsRef<str>,
{
    // We can pick any unsigned integer type > vocab_size;
    // See [`wordchipper::types::TokenType`].
    type T = u32;
    type K = String;
    type C = u64;

    let options = BinaryPairVocabTrainerOptions::new(
        OA_GPT3_CL100K_WORD_PATTERN,
        vocab_size,
    );

    let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();

    for batch in batches {
        // The trainer has no parallelism.
        // The perceived benefits of parallelism in the trainer
        // are insignificant if the IO for the sample source is
        // fed by another thread.
        trainer.update_from_samples(batch.as_ref());
    }

    let vocab: UnifiedTokenVocab<T> = trainer
        .train(Default::default())
        .expect("training failed");

    if let Some(path) = tiktoken_save_path {
        save_span_map_to_tiktoken_path(&vocab.span_vocab.span_map(), &path)
            .expect("failed to save tiktoken vocab");
        println!("- tiktoken vocab: {path:?}");
    }

    let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::init(vocab.clone(), None);
    let encoder = ParallelRayonEncoder::new(encoder);

    let decoder = DefaultTokenDecoder::from_unified_vocab(vocab.clone());
    let decoder = ParallelRayonDecoder::new(decoder);
}
```

#### Running `examples/tokenizer_trainer`

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

