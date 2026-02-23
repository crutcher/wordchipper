# WordChipper Encode/Decode Side-by-Side Benchmarks

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
