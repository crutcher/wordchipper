#![allow(missing_docs)]

//! Profiling investigation for issue #173:
//! Why is multilingual text disproportionately slower for wordchipper encoding?

use std::sync::Arc;
use std::time::Instant;

use wordchipper::{
    TokenEncoder,
    TokenEncoderBuilder,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    pretrained::openai::OATokenizer,
    spanning::{SpanRef, TextSpanner, TextSpannerBuilder},
};

static DIVERSE_CORPUS: &str = include_str!("../benches/data/multilingual.txt");
static ENGLISH_CORPUS: &str = include_str!("../benches/data/english.txt");

struct SpanStats {
    total_spans: usize,
    vocab_hits: usize,
    compound_spans: usize,
    total_compound_bytes: usize,
    max_compound_bytes: usize,
    total_bytes: usize,
}

fn collect_span_stats(
    spanner: &Arc<dyn TextSpanner>,
    vocab: &UnifiedTokenVocab<u32>,
    text: &str,
) -> SpanStats {
    let mut stats = SpanStats {
        total_spans: 0,
        vocab_hits: 0,
        compound_spans: 0,
        total_compound_bytes: 0,
        max_compound_bytes: 0,
        total_bytes: text.len(),
    };

    spanner.for_each_split_span(text, &mut |span_ref| {
        if let SpanRef::Word(range) = span_ref {
            stats.total_spans += 1;
            let span = &text[range.clone()].as_bytes();
            if vocab.lookup_token(span).is_some() {
                stats.vocab_hits += 1;
            } else {
                stats.compound_spans += 1;
                let len = range.len();
                stats.total_compound_bytes += len;
                if len > stats.max_compound_bytes {
                    stats.max_compound_bytes = len;
                }
            }
        }
        true
    });

    stats
}

fn time_spanning(
    spanner: &Arc<dyn TextSpanner>,
    text: &str,
    iterations: usize,
) -> std::time::Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        spanner.for_each_split_span(std::hint::black_box(text), &mut |_| true);
    }
    start.elapsed() / iterations as u32
}

fn time_encoding(
    encoder: &Arc<dyn TokenEncoder<u32>>,
    text: &str,
    iterations: usize,
) -> (std::time::Duration, usize) {
    let tokens = encoder.try_encode(text).unwrap();
    let token_count = tokens.len();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encoder.try_encode(std::hint::black_box(text)).unwrap();
    }
    (start.elapsed() / iterations as u32, token_count)
}

fn run_profile(model: OATokenizer, model_name: &str, text: &str, corpus_name: &str) {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: Arc<UnifiedTokenVocab<u32>> = model.load_vocab(&mut disk_cache).unwrap().into();

    let spanner = TextSpannerBuilder::from_vocab(&vocab)
        .with_parallel(false)
        .build();

    let encoder = TokenEncoderBuilder::new(vocab.clone())
        .with_parallel(false)
        .build();

    let text_10x = text.repeat(10);

    let iterations = 50;

    // Collect span statistics
    let stats = collect_span_stats(&spanner, &vocab, &text_10x);

    // Time spanning only
    let span_time = time_spanning(&spanner, &text_10x, iterations);

    // Time full encoding
    let (encode_time, token_count) = time_encoding(&encoder, &text_10x, iterations);

    // Derive BPE time
    let bpe_time = encode_time.saturating_sub(span_time);

    let bytes = text_10x.len();
    let span_mbps = bytes as f64 / span_time.as_secs_f64() / 1_000_000.0;
    let encode_mbps = bytes as f64 / encode_time.as_secs_f64() / 1_000_000.0;
    let bpe_mbps = if bpe_time.as_nanos() > 0 {
        bytes as f64 / bpe_time.as_secs_f64() / 1_000_000.0
    } else {
        f64::INFINITY
    };

    let avg_compound = if stats.compound_spans > 0 {
        stats.total_compound_bytes as f64 / stats.compound_spans as f64
    } else {
        0.0
    };

    let vocab_hit_pct = if stats.total_spans > 0 {
        100.0 * stats.vocab_hits as f64 / stats.total_spans as f64
    } else {
        0.0
    };

    let bytes_per_token = if token_count > 0 {
        bytes as f64 / token_count as f64
    } else {
        0.0
    };

    println!("=== {model_name} / {corpus_name} ===");
    println!("  Input: {} bytes, {} tokens ({:.2} bytes/token)", bytes, token_count, bytes_per_token);
    println!("  Spans: {} total, {} vocab hits ({:.1}%), {} compound",
        stats.total_spans, stats.vocab_hits, vocab_hit_pct, stats.compound_spans);
    println!("  Compound: avg {:.1} bytes, max {} bytes, {:.1}% of input bytes",
        avg_compound, stats.max_compound_bytes,
        100.0 * stats.total_compound_bytes as f64 / stats.total_bytes as f64);
    println!("  Spanning: {:.2} ms ({:.1} MB/s)", span_time.as_secs_f64() * 1000.0, span_mbps);
    println!("  Encoding: {:.2} ms ({:.1} MB/s)", encode_time.as_secs_f64() * 1000.0, encode_mbps);
    println!("  BPE only: {:.2} ms ({:.1} MB/s)", bpe_time.as_secs_f64() * 1000.0, bpe_mbps);
    println!("  BPE fraction: {:.1}% of total encode time",
        100.0 * bpe_time.as_secs_f64() / encode_time.as_secs_f64());
    println!();
}

fn analyze_bpe_cost(
    spanner: &Arc<dyn TextSpanner>,
    vocab: &Arc<UnifiedTokenVocab<u32>>,
    text: &str,
    model_name: &str,
    corpus_name: &str,
) {
    let mut compound_lengths: Vec<usize> = Vec::new();
    let mut merge_iterations: Vec<usize> = Vec::new();

    spanner.for_each_split_span(text, &mut |span_ref| {
        if let SpanRef::Word(range) = span_ref {
            let span = &text[range.clone()].as_bytes();
            if vocab.lookup_token(span).is_none() {
                let byte_count = span.len();
                compound_lengths.push(byte_count);

                // Simulate the BPE merge loop to count iterations
                let mut tokens: Vec<u32> = Vec::with_capacity(byte_count);
                vocab.byte_vocab().append_tokens(span, &mut tokens);
                let mut iters = 0;
                let start = 0;
                let stop = start + 2;
                while tokens.len() >= stop {
                    if let Some((token, idx)) = tokens[start..]
                        .windows(2)
                        .enumerate()
                        .filter_map(|(idx, w)| {
                            vocab.lookup_pair(&(w[0], w[1])).map(|token| (token, idx))
                        })
                        .min()
                    {
                        let idx = start + idx;
                        tokens[idx] = token;
                        tokens.remove(idx + 1);
                        iters += 1;
                    } else {
                        break;
                    }
                }
                merge_iterations.push(iters);
            }
        }
        true
    });

    if compound_lengths.is_empty() {
        println!("  [{model_name}/{corpus_name}] No compound spans");
        return;
    }

    compound_lengths.sort();
    merge_iterations.sort();

    let n = compound_lengths.len();
    let total_bytes: usize = compound_lengths.iter().sum();
    let total_merges: usize = merge_iterations.iter().sum();

    // Estimate total pair lookups: each merge iteration scans (current_len - 1) pairs
    // For a span of initial length L, iteration i scans (L - i - 1) pairs
    // This sums to approximately L*(L-1)/2 for worst case
    // Better estimate: sum over spans
    let mut total_pair_lookups: usize = 0;
    for (len, merges) in compound_lengths.iter().zip(merge_iterations.iter()) {
        // Each iteration scans remaining - 1 pairs. Over `merges` iterations:
        // sum from i=0..merges of (len - i - 1) = merges*(len-1) - merges*(merges-1)/2
        let m = *merges;
        let l = *len;
        if l > 0 && m > 0 {
            total_pair_lookups += m * (l - 1) - m * (m - 1) / 2;
        }
    }

    let p50_len = compound_lengths[n / 2];
    let p90_len = compound_lengths[n * 9 / 10];
    let p99_len = compound_lengths[n * 99 / 100];
    let max_len = *compound_lengths.last().unwrap();

    let p50_merges = merge_iterations[n / 2];
    let p90_merges = merge_iterations[n * 9 / 10];
    let max_merges = *merge_iterations.last().unwrap();

    println!("  [{model_name}/{corpus_name}] BPE detail:");
    println!("    Compound spans: {n}");
    println!("    Byte lengths: avg={:.1}, p50={}, p90={}, p99={}, max={}",
        total_bytes as f64 / n as f64, p50_len, p90_len, p99_len, max_len);
    println!("    Merge iters:  avg={:.1}, p50={}, p90={}, max={}",
        total_merges as f64 / n as f64, p50_merges, p90_merges, max_merges);
    println!("    Total merge iterations: {total_merges}");
    println!("    Est. total pair lookups: {total_pair_lookups}");

    // Show distribution of compound span lengths
    let mut buckets = [0usize; 8]; // 1-2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129+
    for &l in &compound_lengths {
        let bucket = match l {
            1..=2 => 0,
            3..=4 => 1,
            5..=8 => 2,
            9..=16 => 3,
            17..=32 => 4,
            33..=64 => 5,
            65..=128 => 6,
            _ => 7,
        };
        buckets[bucket] += 1;
    }
    let labels = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+"];
    print!("    Length distribution: ");
    for (label, count) in labels.iter().zip(buckets.iter()) {
        if *count > 0 {
            print!("{label}:{count}  ");
        }
    }
    println!();
    println!();
}

#[test]
#[ignore]
fn profile_multilingual_slowdown() {
    println!();
    println!("======================================================");
    println!("  Multilingual encoding slowdown investigation (#173)");
    println!("======================================================");
    println!();

    for model in [OATokenizer::Cl100kBase, OATokenizer::O200kBase] {
        let name = format!("{:?}", model);
        run_profile(model, &name, ENGLISH_CORPUS, "English");
        run_profile(model, &name, DIVERSE_CORPUS, "Multilingual");
    }

    println!("======================================================");
    println!("  Detailed BPE merge analysis");
    println!("======================================================");
    println!();

    for model in [OATokenizer::Cl100kBase, OATokenizer::O200kBase] {
        let name = format!("{:?}", model);
        let mut disk_cache = WordchipperDiskCache::default();
        let vocab: Arc<UnifiedTokenVocab<u32>> = model.load_vocab(&mut disk_cache).unwrap().into();
        let spanner = TextSpannerBuilder::from_vocab(&vocab)
            .with_parallel(false)
            .build();

        let english_10x = ENGLISH_CORPUS.repeat(10);
        let diverse_10x = DIVERSE_CORPUS.repeat(10);

        analyze_bpe_cost(&spanner, &vocab, &english_10x, &name, "English");
        analyze_bpe_cost(&spanner, &vocab, &diverse_10x, &name, "Multilingual");
    }
}
