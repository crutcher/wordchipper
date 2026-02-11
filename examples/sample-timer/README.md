# WordChipper Encode/Decode Side-by-Side Benchmarks

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
