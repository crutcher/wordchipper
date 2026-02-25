| Field        | Value            |
| ------------ | ---------------- |
| **Date**     | `2026-02-22`     |
| **Commit**   | `88df0b7` (main) |
| **Hardware** | Apple M4 Pro     |

## Encoder Variants

- **buffer_sweep** - O(n^2) reference implementation using a separate working buffer
- **tail_sweep** - O(n^2) linear-scan BPE merge using the output buffer tail as working memory
- **merge_heap** - O(n^2) with parallel pair-rank tracking
- **priority_merge** - O(n log n) binary min-heap with doubly-linked list (default)

## Single-String Encoding (median MB/s)

Corpus: `english.txt` (~7 KB) and `multilingual.txt` (~9 KB), repeated 10x.

| Encoder            | diverse cl100k | diverse o200k | english cl100k | english o200k |
| ------------------ | -------------- | ------------- | -------------- | ------------- |
| buffer_sweep       | 56             | 27            | 86             | 81            |
| tail_sweep         | 57             | 28            | 88             | 86            |
| merge_heap         | 63             | 38            | 99             | 95            |
| **priority_merge** | **92**         | **84**        | **125**        | **119**       |
| tiktoken-rs        | 11             | 11            | 11             | 11            |
| HF tokenizers      | 6              | 6             | 6              | 6             |

## Parallel Batch Encoding (median MB/s)

Corpus: 1024 samples from fineweb-edu shard 0 (~4.2 MB batch). All engines use rayon `par_iter()`.

| Encoder        | cl100k | o200k |
| -------------- | ------ | ----- |
| buffer_sweep   | 1,443  | 1,212 |
| tail_sweep     | 1,417  | 1,163 |
| merge_heap     | 1,455  | 1,343 |
| priority_merge | 961    | 930   |
| tiktoken-rs    | 163    | 144   |
| HF tokenizers  | 9      | 9     |

## Spanning (median MB/s)

| Spanner      | english | diverse |
| ------------ | ------- | ------- |
| cl100k logos | 467     | 378     |
| o200k logos  | 471     | 426     |
| cl100k regex | 20      | 21      |
| o200k regex  | 13      | 14      |
