use arrow::array::StringArray;
use clap::Parser;
use clap::builder::PossibleValuesParser;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use similar::{ChangeTag, TextDiff};
use std::iter::Iterator;
use std::time::Duration;
use strum::IntoEnumIterator;
use tiktoken_rs::CoreBPE;
use wordchipper::compat::slices::{inner_slice_view, inner_str_view};
use wordchipper::compat::timers;
use wordchipper::decoders::{DefaultTokenDecoder, TokenDecoder};
use wordchipper::disk_cache::WordchipperDiskCache;
use wordchipper::encoders::{DefaultTokenEncoder, TokenEncoder};
use wordchipper::pretrained::openai::OATokenizer;
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper_data::dataset::DatasetCacheConfig;

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
fn build_oatokenizer_parser() -> PossibleValuesParser {
    static OATOKENIZER_VARIANTS: Lazy<Vec<String>> =
        Lazy::new(|| OATokenizer::iter().map(|v| v.to_string()).collect());

    PossibleValuesParser::new(
        &*OATOKENIZER_VARIANTS
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
    )
}

/// Example encoders trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to sample shard dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// The pretrained model to compare.
    #[arg(
        long,
        value_parser = build_oatokenizer_parser(),
        default_value_t = OATokenizer::O200kHarmony
    )]
    pub model: OATokenizer,

    /// The shards to use for timing.
    #[arg(long, default_values_t = vec![0, 1])]
    pub shards: Vec<usize>,

    /// The batch size to use for timing.
    #[arg(long, default_value_t = 512)]
    pub batch_size: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    run(&args)?;

    Ok(())
}

#[allow(unused)]
fn run(args: &Args) -> anyhow::Result<()> {
    type T = u32;

    let mut shard_data_cache = DatasetCacheConfig::default()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    println!("Model: {}", args.model);

    let tt_bpe = load_tiktoken(args.model)?;

    let vocab: UnifiedTokenVocab<T> = {
        let mut disk_cache = WordchipperDiskCache::default();
        args.model.load(&mut disk_cache)?
    };

    let encoder = wordchipper::concurrency::rayon::ParallelRayonEncoder::new(
        DefaultTokenEncoder::new(vocab.clone(), None),
    );

    let decoder = wordchipper::concurrency::rayon::ParallelRayonDecoder::new(
        DefaultTokenDecoder::from_unified_vocab(vocab.clone()),
    );

    let mut samples = Vec::new();
    {
        for &shard in &args.shards {
            for batch in shard_data_cache.read_batches(shard, true)? {
                let batch = batch?;
                let column = batch
                    .column_by_name("text")
                    .expect("failed to find 'text' column in batch")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();

                for val in column {
                    let val = val.unwrap().to_string();
                    samples.push(val);
                }
            }
        }
    }

    println!();
    println!("Samples Summary:");
    let sample_count = samples.len();
    println!("- count: {}", sample_count);
    let total_sample_bytes = samples.iter().map(|s| s.len()).sum::<usize>();
    println!("- total size: {}", total_sample_bytes);
    let avg_sample_size = total_sample_bytes / sample_count;
    println!("- avg size: {avg_sample_size}");

    let sample_batches: Vec<&[String]> = samples.chunks(args.batch_size).collect::<Vec<_>>();
    let num_batches = sample_batches.len();

    let avg_batch_size_bytes = total_sample_bytes / num_batches;
    println!("- avg batch size bytes: {avg_batch_size_bytes}");

    println!();
    println!("Timing Config:");
    println!("- batch size: {}", args.batch_size);
    println!("- num batches: {}", num_batches);

    println!();
    println!("Timing Encode:");
    let mut wc_token_batches: Vec<Vec<Vec<T>>> = Default::default();
    let mut wc_total_token_count = 0;
    let mut tt_total_token_count = 0;
    let mut wc_batch_durations = vec![];
    let mut tt_batch_durations = vec![];
    for (idx, batch) in sample_batches.iter().enumerate() {
        let str_batch = inner_str_view(batch);

        let (durationn, wc_batch_tokens) =
            timers::timeit(|| encoder.try_encode_batch(&str_batch).unwrap());
        wc_batch_durations.push(durationn);

        wc_total_token_count += wc_batch_tokens
            .iter()
            .map(|tokens| tokens.len())
            .sum::<usize>();

        {
            let (duration, tt_batch_tokens) = timers::timeit(|| {
                str_batch
                    .par_iter()
                    .map(|s| tt_bpe.encode_with_special_tokens(s))
                    .collect::<Vec<_>>()
            });
            tt_batch_durations.push(duration);

            tt_total_token_count += tt_batch_tokens
                .iter()
                .map(|tokens| tokens.len())
                .sum::<usize>();

            assert_eq!(&tt_batch_tokens, &wc_batch_tokens);
        }

        wc_token_batches.push(wc_batch_tokens);
    }

    for (name, durations) in [
        ("wordchipper", &wc_batch_durations),
        ("tiktoken-rs", &tt_batch_durations),
    ] {
        let mean_time = durations.iter().sum::<Duration>() / num_batches as u32;
        println!(
            "- {name}:\t{mean_time:>10.1?}, {:>15}",
            format_bps(avg_batch_size_bytes, mean_time),
        );
    }

    println!();
    println!("Observed Bytes/Token Stats:");
    for (name, token_count) in [
        ("wordchipper", wc_total_token_count),
        ("tiktoken-rs", tt_total_token_count),
    ] {
        println!("- {name} token count: {}", token_count);
        println!(
            "- {name} byte/token: {:.2}",
            total_sample_bytes as f64 / token_count as f64
        );
    }

    println!();
    println!("Timing Decode:");
    let mut wc_batch_decode_durations = vec![];
    let mut tt_batch_decode_durations = vec![];
    for (idx, sample) in sample_batches.iter().enumerate() {
        let batch = &wc_token_batches[idx];

        let expected: Vec<String> = sample
            .par_iter()
            .map(|t| encoder.spanner().remove_gaps(t))
            .collect();

        let slices = inner_slice_view(batch);

        {
            let (duration, wc_decoded) = timers::timeit(|| {
                decoder
                    .try_decode_batch_to_strings(&slices)
                    .unwrap()
                    .unwrap()
            });
            wc_batch_decode_durations.push(duration);

            verify_decode(&wc_decoded, &expected);
        }

        {
            let (duration, tt_decoded) = timers::timeit(|| {
                batch
                    .par_iter()
                    .map(|tokens| tt_bpe.decode(tokens.clone()).unwrap())
                    .collect::<Vec<_>>()
            });

            tt_batch_decode_durations.push(duration);

            verify_decode(&tt_decoded, &expected);
        }
    }

    for (name, durations) in [
        ("wordchipper", &wc_batch_decode_durations),
        ("tiktoken-rs", &tt_batch_decode_durations),
    ] {
        let mean_time = durations.iter().sum::<Duration>() / num_batches as u32;
        println!(
            "- {name}:\t{mean_time:>10.1?}, {:>15}",
            format_bps(avg_batch_size_bytes, mean_time),
        );
    }

    Ok(())
}

pub fn verify_decode(
    actual: &[String],
    expected: &[String],
) {
    let sstrs: Vec<&str> = inner_str_view(expected);
    let dstrs: Vec<&str> = inner_str_view(actual);

    let mismatch: Vec<(&str, &str)> = sstrs
        .iter()
        .zip(dstrs.iter())
        .collect::<Vec<_>>()
        .par_iter()
        .filter(|(s, d)| s != d)
        .map(|&(&s, &d)| (s, d))
        .collect();

    if let Some((s, d)) = mismatch.into_iter().next() {
        let diff = TextDiff::from_lines(s, d);

        for change in diff.iter_all_changes() {
            let sign = match change.tag() {
                ChangeTag::Delete => "-",
                ChangeTag::Insert => "+",
                ChangeTag::Equal => " ",
            };
            print!("{}{}", sign, change);
        }
        panic!("MISMATCH");
    }
}
