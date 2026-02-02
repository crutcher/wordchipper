# Example Tokenizer Load Pretrained

Each shard is ~90MB parquet file.

- 128/64 Core Thread Ripper
- _NOTE: there are still some tokenization differences to resolve here._

```terminaloutput
$ RAYON_NUM_THREADS=16 cargo run --release -p token-cli -- --dataset-dir /media/Data/nanochat/dataset 
   Compiling wordchipper v0.1.2 (/home/crutcher/git/wordchipper/crates/wordchipper)
   Compiling token-cli v0.0.0 (/home/crutcher/git/wordchipper/examples/token-cli)
    Finished `release` profile [optimized] target(s) in 1.87s
     Running `target/release/token-cli --dataset-dir /media/Data/nanochat/dataset`

Samples Summary:
- count: 53248
- total size: 254737840
- avg size: 4783
- avg batch size bytes: 2449402

Timing Config:
- batch size: 512
- num batches: 104

Timing Encode:
- wordchipper:      14.6ms,    160.32 MiB/s
- tiktoken-rs:      32.7ms,     71.33 MiB/s

Observed Bytes/Token Stats:
- wordchipper token count: 54749669
- wordchipper byte/token: 4.65
- tiktoken-rs token count: 53251930
- tiktoken-rs byte/token: 4.78

Timing Decode:
- wordchipper:       2.8ms,    840.54 MiB/s
- tiktoken-rs:       2.1ms,      1.08 GiB/s
```
