# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml)
[![license](https://shields.io/badge/license-MIT-blue)](LICENSE)

I am usually available as `@crutcher` on the Burn Discord:

* Burn Discord: [![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)

## Crates

- [wordchipper](crates/wordchipper) - HPC BPE Tokenizer; soon to be extracted into its own crate.
- [wordchipper-data](crates/wordchipper-data) - Nanochat Data Shard Pull / Cache lib.

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

It is inspired by [nanochat rustbpe](https://github.com/karpathy/nanochat/tree/master/rustbpe)

The current status is productionization towards an alpha release.

TODO:

- New Name / New Repo. ( `wordchipper` conflicts, alas)
- Pluggable URL/Local File Cache
    - PreTrained
    - TrainingShards
- Benchmarks.
- More complete Error handling (as `Result`s, not panics).
- Tuning
    - Instrument `tiktoken` (via `tracing`).
    - Compare / fix perf differences.
- Python/C*/Java Bindings?

See:

- [examples/tokenizer_trainer](crates/wordchipper/examples/tokenizer_trainer)

## Crate Features

#### feature: ``default``

* ``client``

The default feature enables ``client``, with no training code.

#### feature: ``client``

* ``ahash``
* ``rayon``
* ``std``

The default client is focused on loading vocabularies and running
high performance encoders / decoders.

#### feature: ``training``

* ``compact_str``
* ``dary_heap``
* ``std``

The training feature enables the training code.

#### feature: ``std`` / ``no_std``

The ``std`` feature enables the use of the ``std`` library;
and the ``no_std`` feature enables deps needed when ``std`` is not enabled.
(Negative feature deps are not stable yet.)

Note: I am unsure if this is complete. It is tested CI, but I'm unsure
if I've fully covered it; and I haven't worked out a ``no_std`` deploy test yet.

#### feature: ``ahash``

This swaps all HashMap/HashSet implementations for ``ahash`; which is a performance
win on many/(most?) modern CPUs.

This is done by the ``types::hash_types::CommonHash{*}`` type alias machinery.
See also the ``hashbrown`` dep used by ``no_std``.

#### feature: ``rayon``

This enables some parallelism wrappers using the ``rayon`` crate.

TODO: I intend on providing a ``tokio`` based ``async`` parallelism mechanism
as well, to structure more direct ``regex`>`encode`` pipeline parallelism.

#### feature: ``tracing``

This enables a number of ``tracing`` instrumentation points.
This is only useful for timing tracing of the library itself.

## Client Usage

### UnifiedTokenVocab

The `UnifiedTokenVocab` is a unified representation of the vocabularies
used by the `TokenEncoder` and `TokenDecoder` clients. It contains:

* `SegmentationConfig` - describing the span/word regex and the special token map.
* `ByteMapVocab` - describing the `{ u8 -> T }` mapping.
* `SpanMapVocab` - describing the `{ Vec<u8> -> T }` mapping.
* `PairMapVocab` - describing known `{ (T, T) -> T }` merge pairs.

#### Loading Pretrained Vocabularies

This is only partially implemented; it still requires a fair amount of manual work.

A collection of metadata about known pretrained vocabularies is available:

* `wordchipper::vocab::public`

What is incomplete is a local URL cache plus workflow for assembling a vocab
from the known metadata.

A loading example exists in the `examples/token-cli` crate.

```rust,no_run
use wordchipper::decoders::{DictionaryDecoder, TokenDecoder};
use wordchipper::encoders::{MergeHeapVocabEncoder, TokenEncoder};
use wordchipper::rayon::{ParallelRayonDecoder, ParallelRayonEncoder};
use wordchipper::regex::{regex_pool_supplier, RegexWrapperPattern};
use wordchipper::segmentation::{SegmentationConfig, TextSegmentor};
use wordchipper::vocab::io::load_tiktoken_vocab_path;
use wordchipper::vocab::public::openai::{oa_gpt2_r50k_specials, OA_GPT2_R50K_BASE_TIKTOKEN, OA_GPT2_R50K_WORD_PATTERN};
use wordchipper::vocab::UnifiedTokenVocab;

type T = u32;

let pattern: RegexWrapperPattern = OA_GPT2_R50K_WORD_PATTERN.into();

let r50k_tiktoken = OA_GPT2_R50K_BASE_TIKTOKEN;
// If we had a download cache, we'd use OA_GPT_R50K_BASE_TIKTOKEN.url here:
let span_map = load_tiktoken_vocab_path(tokenizer_file)?;

let segmentation = SegmentationConfig::<T>::from_pattern(pattern.clone()).with_special_words(
    oa_gpt2_r50k_specials()
        .iter()
        .map(|(s, t)| (s, T::from_usize(*t).unwrap())),
);

let vocab: Arc<UnifiedTokenVocab<T>> =
    UnifiedTokenVocab::from_span_vocab(segmentation, span_map.into()).into();

let encoder: MergeHeapVocabEncoder<T> =
    MergeHeapVocabEncoder::<T>::init_with_factory(vocab.clone(), regex_pool_supplier);
let encoder = ParallelRayonEncoder::new(encoder);

let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
let decoder = ParallelRayonDecoder::new(decoder);
```

### TokenEncoder Clients

Encoder clients should use:

* `MergeHeapVocabEncoder` - the current default (only?) `TokenEncoder`.
* `ParallelRayonEncoder` - a batch parallelism wrapper around any `TokenEncoder`.

```rust,no_run
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::encoders::MergeHeapVocabEncoder;
use wordchipper::encoders::TokenEncoder;
use wordchipper::types::TokenType;
use std::sync::Arc;

fn example<T: TokenType>(
    vocab: Arc<UnifiedTokenVocab<T>>,
    batch: &[String],
) -> Vec<Vec<T>> {
    let encoder: MergeHeapVocabEncoder<T> = MergeHeapVocabEncoder::init(vocab);

    #[cfg(feature = "rayon")]
    let encoder = wordchipper::rayon::ParallelRayonEncoder::new(encoder);

    encoder.try_encode_batch(batch).unwrap()
}
```

### TokenDecoder Clients

Decoder clients should use:

* `DictionaryDecoder` - the fastest `TokenDecoder`.
* `ParallelRayonDecoder` - a batch parallelism wrapper around any `TokenDecoder`.

```rust,no_run
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::decoders::DictionaryDecoder;
use wordchipper::decoders::TokenDecoder;
use wordchipper::types::TokenType;
use std::sync::Arc;

fn example<T: TokenType>(
    vocab: Arc<UnifiedTokenVocab<T>>,
    batch: &[Vec<T>],
) -> Vec<String> {
    let decoder: DictionaryDecoder<T> = DictionaryDecoder::from_unified_vocab(vocab);

    #[cfg(feature = "rayon")]
    let decoder = wordchipper::rayon::ParallelRayonDecoder::new(decoder);

    decoder.try_decode_batch_to_strings(batch).unwrap()
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
use wordchipper::vocab::io::save_tiktoken_vocab_path;
use wordchipper::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
use wordchipper::vocab::{ByteMapVocab, UnifiedTokenVocab};
use wordchipper::encoders::MergeHeapVocabEncoder;
use wordchipper::decoders::DictionaryDecoder;
use wordchipper::rayon::{ParallelRayonEncoder, ParallelRayonDecoder};
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

    let byte_vocab: Arc<ByteMapVocab<T>> = Arc::new(Default::default());

    let vocab: Arc<UnifiedTokenVocab<T>> = trainer
        .train(byte_vocab.clone())
        .expect("training failed")
        .into();

    if let Some(path) = tiktoken_save_path {
        save_tiktoken_vocab_path(&vocab.span_vocab.span_map(), &path)
            .expect("failed to save tiktoken vocab");
        println!("- tiktoken vocab: {path:?}");
    }

    let encoder: MergeHeapVocabEncoder<T> = MergeHeapVocabEncoder::init(vocab.clone());
    let encoder = ParallelRayonEncoder::new(encoder);

    let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
    let decoder = ParallelRayonDecoder::new(decoder);
}
```

#### Running `examples/tokenizer_trainer`

Each shard is ~90MB parquet file.

- 64 Core AMD

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir /media/Data/nanochat/dataset --time-encode-decode
   Compiling tokenizer_trainer v0.0.0 (/home/crutcher/git/brn-nanochat/crates/wordchipper/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 24.15s
     Running `target/release/tokenizer_trainer --dataset-dir /media/Data/nanochat/dataset --time-encode-decode`
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
- training_duration: 188.60s
- vocab_size: 65535

Samples Summary:
- count: 20480
- avg size: 4741

Timing Config:
- batch size: 512

Timing Encode:
- batch avg: 71.640541ms
- sample avg: 139.922µs
- avg bps: 33.88 MB/s

Observed Bytes/Token Stats:
- total bytes: 97103222
- total tokens: 24645141
- sample byte/token: 3.94

Timing Decode:
- batch avg: 1.801906ms
- sample avg: 3.519µs

real    3m36.164s
user    10m34.735s
sys     0m43.929s
```

## Acknowledgements

* Thank you to [@karpathy](https://github.com/karpathy) and [nanochat](https://github.com/karpathy/nanochat)
  for the work on `rustbpe`.
* Thank you to [tiktoken](https://github.com/openai/tiktoken) for their initial work in the rust tokenizer space.

