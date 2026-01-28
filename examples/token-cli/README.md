# Example Tokenizer Load Pretrained

Each shard is ~90MB parquet file.

- 128/64 Core Thread Ripper

```terminaloutput
$ time cargo run --release -p token-cli -- --dataset-dir /media/Data/nanochat/dataset load 
   Compiling wordchipper-disk-cache v0.0.10 (/home/crutcher/git/wordchipper/crates/wordchipper-disk-cache)
   Compiling wordchipper v0.0.10 (/home/crutcher/git/wordchipper/crates/wordchipper)
   Compiling token-cli v0.0.0 (/home/crutcher/git/wordchipper/crates/wordchipper/examples/token-cli)
    Finished `release` profile [optimized] target(s) in 1.80s
     Running `target/release/token-cli --dataset-dir /media/Data/nanochat/dataset load`
Loading Shards: [0]
...

Samples Summary:
- count: 20480
- avg size: 4741

Timing Config:
- batch size: 512
- num batches: 40

Timing Encode:
- batch avg: 59.67926ms
- sample avg: 116.561µs
- avg bps: 40.67 MB/s

Observed Bytes/Token Stats:
- total bytes: 97103222
- total tokens: 20854908
- sample byte/token: 4.66

Timing Decode:
- batch avg: 2.492366ms
- sample avg: 4.867µs
```
