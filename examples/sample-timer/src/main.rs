use std::{io, io::IsTerminal, iter::Iterator, time::Duration};

use anyhow::bail;
use arrow::array::{Array, StringArray};
use clap::{
    Parser,
    builder::{PossibleValuesParser, TypedValueParser},
};
use indicatif::ProgressBar;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use strum::IntoEnumIterator;
use tiktoken_rs::{CoreBPE, Rank};
use wordchipper::{
    compat::{
        slices::{inner_slice_view, inner_str_view},
        timers::timeit,
    },
    concurrency::rayon::{ParallelRayonDecoder, ParallelRayonEncoder},
    decoders::{DefaultTokenDecoder, TokenDecoder},
    disk_cache::WordchipperDiskCache,
    encoders::{DefaultTokenEncoder, TokenEncoder},
    pretrained::openai::OATokenizer,
    vocab::UnifiedTokenVocab,
};
use wordchipper_data::dataset::{DatasetCache, DatasetCacheConfig};

/// Side-by-Side Wordchipper vs Tiktoken Benchmark.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to sample shard dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// The shards to use for timing.
    #[arg(long, default_values_t = vec![0, 1, 2, 3])]
    pub shards: Vec<usize>,

    /// The batch size to use for timing.
    #[arg(long, default_value_t = 512)]
    pub batch_size: usize,

    /// The pretrained model to compare.
    #[arg(
        long,
        value_parser = build_tokenizer_parser(),
        default_value = "oa:o200k_harmony",
    )]
    pub model: OATokenizer,
}

#[allow(unused)]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let display_progress = io::stdout().is_terminal();

    let mut shard_data_cache = DatasetCacheConfig::default()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    println!("Model: \"oa:{}\"", args.model);
    println!("- shards: {:?}", args.shards);
    println!("- batch_size: {}", args.batch_size);

    let tt_enc_dec = load_tiktoken(args.model)?;

    let vocab: UnifiedTokenVocab<Rank> = {
        let mut disk_cache = WordchipperDiskCache::default();
        args.model.load(&mut disk_cache)?
    };
    let wc_encoder = ParallelRayonEncoder::new(DefaultTokenEncoder::new(vocab.clone(), None));
    let wc_decoder =
        ParallelRayonDecoder::new(DefaultTokenDecoder::from_unified_vocab(vocab.clone()));

    let mut stats = Vec::new();

    for_each_batch(
        display_progress,
        &args.shards,
        args.batch_size,
        &mut shard_data_cache,
        &mut |str_batch: &[&str]| -> anyhow::Result<()> {
            let (wc_enc_duration, wc_batch_tokens) =
                timeit(|| wc_encoder.try_encode_batch(str_batch).unwrap());

            let (tt_enc_duration, tt_batch_tokens) = timeit(|| {
                str_batch
                    .par_iter()
                    .map(|s| tt_enc_dec.encode_with_special_tokens(s))
                    .collect::<Vec<_>>()
            });

            verify_encode(str_batch, &wc_batch_tokens, &tt_batch_tokens)?;

            let batch_tokens: Vec<&[Rank]> = inner_slice_view(&wc_batch_tokens);

            let (wc_dec_duration, wc_batch_decode) = timeit(|| {
                wc_decoder
                    .try_decode_batch_to_strings(&batch_tokens)
                    .unwrap()
                    .unwrap()
            });

            let (tt_dec_duration, tt_batch_decode) = timeit(|| {
                batch_tokens
                    .par_iter()
                    .map(|tokens| tt_enc_dec.decode(tokens.to_vec()).unwrap())
                    .collect::<Vec<_>>()
            });

            // We don't expect the samples to be faithfully decoded,
            // we expect the gap-less spanner stream to match.
            let expected = wc_encoder.spanner().batch_remove_gaps(str_batch);

            verify_decode(&batch_tokens, &expected, &wc_batch_decode, &tt_batch_decode)?;

            stats.push(BatchStats {
                sample_bytes: str_batch.iter().map(|s| s.len()).collect(),
                tokens: wc_batch_tokens.iter().map(|t| t.len()).collect(),
                wc_enc_time: wc_enc_duration,
                tt_enc_time: tt_enc_duration,
                wc_dec_time: wc_dec_duration,
                tt_dec_time: tt_dec_duration,
            });

            Ok(())
        },
    );

    println!();
    println!("Samples Summary:");
    let num_batches = stats.len();
    println!("- num batches: {}", num_batches);

    let avg_batch_bytes = stats.iter().map(|s| s.batch_bytes()).sum::<usize>() / num_batches;
    let avg_sample_bytes = stats.iter().map(|s| s.avg_sample_bytes()).sum::<usize>() / num_batches;
    println!("- avg bytes/sample: {avg_sample_bytes}");

    let total_bytes = stats.iter().map(|s| s.batch_bytes()).sum::<usize>();
    let total_tokens = stats.iter().map(|s| s.total_tokens()).sum::<usize>();
    let avg_bytes_per_token = total_bytes as f64 / total_tokens as f64;
    println!("- avg bytes/token: {avg_bytes_per_token:.1}");

    fn print_timing(
        name: &str,
        batch_time: Duration,
        batch_bytes: usize,
        batch_size: usize,
    ) {
        println!("- {name}");
        println!("  - batch:  {:>10.1?}", batch_time);
        println!("  - sample: {:>10.1?}", batch_time / batch_size as u32);
        println!("  - bps:    {:>10}", format_bps(batch_bytes, batch_time));
    }

    println!();
    println!("Encoder Times:");
    print_timing(
        "wordchipper",
        stats.iter().map(|s| s.wc_enc_time).sum::<Duration>() / num_batches as u32,
        avg_batch_bytes,
        args.batch_size,
    );
    print_timing(
        "tiktoken",
        stats.iter().map(|s| s.tt_enc_time).sum::<Duration>() / num_batches as u32,
        avg_batch_bytes,
        args.batch_size,
    );

    println!();
    println!("Decoder Times:");
    print_timing(
        "wordchipper",
        stats.iter().map(|s| s.wc_dec_time).sum::<Duration>() / num_batches as u32,
        avg_batch_bytes,
        args.batch_size,
    );
    print_timing(
        "tiktoken",
        stats.iter().map(|s| s.tt_dec_time).sum::<Duration>() / num_batches as u32,
        avg_batch_bytes,
        args.batch_size,
    );

    Ok(())
}

fn for_each_batch(
    display_progress: bool,
    shards: &[usize],
    batch_size: usize,
    shard_data_cache: &mut DatasetCache,
    observe_batch: &mut dyn FnMut(&[&str]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let progress_bar = if display_progress {
        ProgressBar::new_spinner()
    } else {
        ProgressBar::hidden()
    };

    let mut batch_count = 0;
    let mut sample_buffer = Vec::new();
    for &shard in shards {
        progress_bar.set_message(format!("Loading shard: {}", shard));
        shard_data_cache.get_shard(shard, true)?;

        for batch in shard_data_cache.read_batches(shard, false)? {
            let batch = batch?;
            let column = batch
                .column_by_name("text")
                .expect("failed to find 'text' column in batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for val in column {
                let val = val.unwrap().to_string();
                sample_buffer.push(val);
            }

            if sample_buffer.len() < batch_size {
                continue;
            }

            let batch = sample_buffer.drain(..batch_size).collect::<Vec<_>>();
            let str_batch = inner_str_view(&batch);

            batch_count += 1;
            progress_bar.set_message(format!("Timing batch: {}", batch_count + 1));
            progress_bar.tick();

            observe_batch(&str_batch)?;
        }
    }

    Ok(())
}

pub fn verify_encode(
    source_batch: &[&str],
    wc_batch: &[Vec<Rank>],
    tt_batch: &[Vec<Rank>],
) -> anyhow::Result<()> {
    assert_eq!(source_batch.len(), wc_batch.len());
    assert_eq!(source_batch.len(), tt_batch.len());
    for (i, s) in source_batch.iter().enumerate() {
        let wc_tokens = &wc_batch[i];
        let tt_tokens = &tt_batch[i];

        if wc_tokens == tt_tokens {
            continue;
        }

        bail!(
            "ENCODER MISMATCH:\n- source: {}\n- wc: {:?}\n- tt: {:?}",
            s,
            wc_tokens,
            tt_tokens,
        )
    }
    Ok(())
}

pub fn verify_decode(
    batch_tokens: &[&[Rank]],
    batch_expected: &[String],
    wc_batch_decode: &[String],
    tt_batch_decode: &[String],
) -> anyhow::Result<()> {
    assert_eq!(batch_tokens.len(), batch_expected.len());
    assert_eq!(batch_tokens.len(), wc_batch_decode.len());
    assert_eq!(batch_tokens.len(), tt_batch_decode.len());

    for (i, expected) in batch_expected.iter().enumerate() {
        let wc_decoded = &wc_batch_decode[i];
        let tt_decoded = &tt_batch_decode[i];

        let wc_match = wc_decoded == expected;
        let tt_match = tt_decoded == expected;

        if !(wc_match || tt_match) {
            let tokens = batch_tokens[i];

            bail!(
                "DECODER MISMATCH:\n- tokens: {:?}\n- expected: {}\n- wc: {}\n- tt: {}",
                tokens,
                expected,
                wc_decoded,
                tt_decoded,
            )
        }
    }

    Ok(())
}

/// Format a bytes/sec string.
pub fn format_bps(
    bytes: usize,
    duration: Duration,
) -> String {
    let bps = bytes as f64 / duration.as_secs_f64();
    format!(r"{}/s", humansize::format_size_i(bps, humansize::BINARY))
}

/// Load a tiktoken model from the given `OATokenizer` enum variant.
fn load_tiktoken(model: OATokenizer) -> anyhow::Result<CoreBPE> {
    use OATokenizer::*;
    match model {
        R50kBase => tiktoken_rs::r50k_base(),
        P50kBase => tiktoken_rs::p50k_base(),
        P50kEdit => tiktoken_rs::p50k_edit(),
        Cl100kBase => tiktoken_rs::cl100k_base(),
        O200kBase => tiktoken_rs::o200k_base(),
        O200kHarmony => tiktoken_rs::o200k_harmony(),
        _ => panic!("unsupported model: {:?}", model),
    }
}

/// Build a clap parser for the `OATokenizer` enum variants.
///
/// The hack here is to get the strum enum variants to list in the clap help.
fn build_tokenizer_parser() -> impl TypedValueParser {
    static OATOKENIZER_VARIANTS: Lazy<Vec<String>> =
        Lazy::new(|| OATokenizer::iter().map(|v| format!("oa:{v}")).collect());

    PossibleValuesParser::new(
        &*OATOKENIZER_VARIANTS
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
    )
    .map(|s| s[3..].parse::<OATokenizer>().unwrap())
}

struct BatchStats {
    pub sample_bytes: Vec<usize>,
    pub tokens: Vec<usize>,

    pub wc_enc_time: Duration,
    pub tt_enc_time: Duration,
    pub wc_dec_time: Duration,
    pub tt_dec_time: Duration,
}

impl BatchStats {
    pub fn len(&self) -> usize {
        self.sample_bytes.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.tokens.iter().sum::<usize>()
    }

    pub fn batch_bytes(&self) -> usize {
        self.sample_bytes.iter().sum::<usize>()
    }

    pub fn avg_sample_bytes(&self) -> usize {
        self.batch_bytes() / self.len()
    }
}
