# Example Tokenizer Trainer

Each shard is ~90MB parquet file.

- 128/64 Core Thread Ripper

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir /media/Data/nanochat/dataset  --time-encode-decode 
   Compiling tokenizer_trainer v0.0.0 (/home/crutcher/git/brn-nanochat/crates/wordchipper/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 1.34s
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
- training_duration: 220.05s
- vocab_size: 65535

Samples Summary:
- count: 20480
- avg size: 4741

Timing Config:
- batch size: 512

Timing Encode:
- batch avg: 69.918721ms
- sample avg: 136.56µs
- avg bps: 34.72 MB/s

Observed Bytes/Token Stats:
- total bytes: 97103222
- total tokens: 24645141
- sample byte/token: 3.94

Timing Decode:
- batch avg: 2.373206ms
- sample avg: 4.635µs

real    3m45.018s
user    78m36.407s
sys     37m53.941s
```
