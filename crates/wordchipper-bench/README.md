# wordchipper-bench

Benchmarks comparing wordchipper's `SpanEncoder` variants against each other and against tiktoken-rs
and HuggingFace tokenizers.

## Benchmarks

| Bench               | What it measures                              |
| ------------------- | --------------------------------------------- |
| `encoding_single`   | Single-string encoding (no parallelism)       |
| `encoding_parallel` | Batch encoding via rayon (`try_encode_batch`) |
| `decoding_single`   | Single-string decoding                        |
| `spanning`          | Text spanning (regex vs logos DFA)            |

### Encoder Variants

- **`incremental_sweep`** - O(n^2) linear-scan BPE merge (current default)
- **`merge_heap`** - O(n^2) with parallel pair-rank tracking
- **`priority_merge`** - O(n log n) binary min-heap with doubly-linked list

## Running

```bash
# All benchmarks
cargo bench -p wordchipper-bench

# Individual benchmarks
cargo bench -p wordchipper-bench --bench encoding_single
cargo bench -p wordchipper-bench --bench encoding_parallel
cargo bench -p wordchipper-bench --bench decoding_single
cargo bench -p wordchipper-bench --bench spanning

# Filter by name
cargo bench -p wordchipper-bench --bench encoding_single -- diverse
cargo bench -p wordchipper-bench --bench encoding_parallel -- priority_merge
```

### Parallel bench data

`encoding_parallel` uses fineweb-edu parquet shards (same as `sample-timer`).
Download shard 0 first:

```bash
cargo run --release -p sample-timer -- \
  --dataset-dir /tmp/wordchipper-bench-data \
  --model openai/cl100k_base --shards 0
```

Then set the data directory (defaults to `/tmp/wordchipper-bench-data`):

```bash
WORDCHIPPER_BENCH_DATA=/tmp/wordchipper-bench-data \
  cargo bench -p wordchipper-bench --bench encoding_parallel
```

## Results

Collected on Apple M4 Pro.

### Single-String Encoding (median MB/s)

Corpus: `english.txt` (~7 KB) and `multilingual.txt` (~9 KB), repeated 10x.

| Encoder            | diverse cl100k | diverse o200k | english cl100k | english o200k |
| ------------------ | -------------- | ------------- | -------------- | ------------- |
| incremental_sweep  | 57             | 28            | 89             | 83            |
| merge_heap         | 63             | 38            | 100            | 98            |
| **priority_merge** | **94**         | **85**        | **130**        | **124**       |
| tiktoken-rs        | 11             | 11            | 11             | 11            |
| HF tokenizers      | 7              | 7             | 7              | 7             |

### Parallel Batch Encoding (median MB/s)

Corpus: 1024 samples from fineweb-edu shard 0 (~4.2 MB batch). All engines use rayon `par_iter()`.

| Encoder           | cl100k | o200k |
| ----------------- | ------ | ----- |
| incremental_sweep | 1,432  | 1,294 |
| merge_heap        | 1,464  | 1,384 |
| priority_merge    | 950    | 919   |
| tiktoken-rs       | 160    | 146   |
| HF tokenizers     | 10     | 9     |

All engines parallelized with rayon. Wordchipper variants are ~8-9x faster than tiktoken-rs
in parallel batch mode on English text.

### Single-String Encoding: `priority_merge` advantage

The `priority_merge` encoder is 3x faster than `incremental_sweep` on diverse/multilingual text
with o200k, where longer multi-byte spans expose the O(n^2) vs O(n log n) gap.
