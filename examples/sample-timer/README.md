# wordchipper vs tiktoken sample timer

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
