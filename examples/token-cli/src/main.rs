use crate::tokenizer_timer::FullMontyTokenizer;
use arrow::array::StringArray;
use clap::Parser;
use similar::{ChangeTag, TextDiff};
use std::time::Duration;
use wordchipper::decoders::{DictionaryDecoder, TokenDecoder};
use wordchipper::disk_cache::WordchipperDiskCache;
use wordchipper::encoders::{DefaultTokenEncoder, TokenEncoder};
use wordchipper::rayon::ParallelRayonDecoder;
use wordchipper::segmentation::TextSegmentor;
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::vocab::public::openai::load_o200k_harmony_vocab;
use wordchipper_data::dataset::DatasetCacheConfig;

mod tokenizer_timer;

/// Example encoders trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Enable verbose output.
    #[arg(long, default_value = "false")]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Parser, Debug)]
pub enum Command {
    /// Load a tokenizer.
    Load {},
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.verbose {
        println!("{:#?}", args);
    }

    match &args.command {
        Some(Command::Load { .. }) => {
            run_load(&args)?;
        }
        None => unreachable!(),
    }

    Ok(())
}

#[allow(unused)]
fn run_load(args: &Args) -> anyhow::Result<()> {
    type T = u32;

    let mut dataset_cache = DatasetCacheConfig::new()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    let tt_bpe = tiktoken_rs::o200k_harmony()?;

    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<T> = load_o200k_harmony_vocab(&mut disk_cache)?;

    let wc_tokenizer = FullMontyTokenizer::init(
        DefaultTokenEncoder::<T>::init(vocab.clone()),
        ParallelRayonDecoder::new(DictionaryDecoder::from_unified_vocab(vocab.clone())),
    );

    let shards: Vec<usize> = vec![0];
    let num_timing_batches = 20;
    let batch_size = 512;

    println!("Loading Shards: {shards:?}");
    println!("...");
    dataset_cache.load_shards(&shards)?;

    let mut samples = Vec::new();
    {
        for batch in dataset_cache
            .read_cached_batches(shards[0])?
            .take(num_timing_batches)
        {
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

    println!();
    println!("Samples Summary:");
    let sample_count = samples.len();
    println!("- count: {}", sample_count);
    let total_sample_bytes = samples.iter().map(|s| s.len()).sum::<usize>();
    let avg_sample_size = total_sample_bytes / sample_count;
    println!("- avg size: {avg_sample_size}");

    let sample_batches: Vec<&[String]> = samples.chunks(batch_size).collect::<Vec<_>>();
    let num_batches = sample_batches.len();

    println!();
    println!("Timing Config:");
    println!("- batch size: {}", batch_size);
    println!("- num batches: {}", num_batches);

    println!();
    println!("Timing Encode:");
    let mut token_batches: Vec<Vec<Vec<T>>> = Vec::with_capacity(sample_batches.len());
    let mut total_wc_token_count = 0;
    let mut total_tt_token_count = 0;

    use rayon::prelude::*;

    let mut wc_batch_times_ns = vec![];
    let mut tt_batch_times_ns = vec![];
    for (idx, batch) in sample_batches.iter().enumerate() {
        let batch = batch.iter().map(|s| s.as_str()).collect::<Vec<_>>();

        let t0 = std::time::Instant::now();
        let wc_encode_batch: anyhow::Result<Vec<Vec<T>>> = batch
            .par_iter()
            .map(|s| wc_tokenizer.encoder.try_encode(s))
            .collect();

        let wc_encode_batch = wc_encode_batch?;

        wc_tokenizer.encoder.try_encode_batch(&batch).unwrap();
        let t1 = std::time::Instant::now();
        let delay = t1.duration_since(t0);
        wc_batch_times_ns.push(delay.as_nanos() as u64);

        total_wc_token_count += wc_encode_batch
            .iter()
            .map(|tokens| tokens.len())
            .sum::<usize>();


        {
            let t0 = std::time::Instant::now();
            let tt_encode_batch = batch
                .par_iter()
                .map(|s| tt_bpe.encode_with_special_tokens(s))
                .collect::<Vec<_>>();
            let t1 = std::time::Instant::now();

            let delay = t1.duration_since(t0);
            tt_batch_times_ns.push(delay.as_nanos() as u64);

            total_tt_token_count += tt_encode_batch
                .iter()
                .map(|tokens| tokens.len())
                .sum::<usize>();
        }

        token_batches.push(wc_encode_batch);
    }

    let avg_batch_time_ns = wc_batch_times_ns.iter().sum::<u64>() / num_batches as u64;
    println!(
        "- wordchipper batch avg: {:#?}",
        Duration::from_nanos(avg_batch_time_ns)
    );
    let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
    println!(
        "- wordchipper sample avg: {:#?}",
        Duration::from_nanos(avg_sample_time_ns)
    );
    let b_p_ns = avg_sample_size as f64 / avg_sample_time_ns as f64;
    let b_p_s = b_p_ns * 1e9;
    let mb_p_s = b_p_s / 1e6;
    println!("- wordchipper avg bps: {:.2} MB/s", mb_p_s);

    {
        let avg_batch_time_ns = tt_batch_times_ns.iter().sum::<u64>() / num_batches as u64;
        println!(
            "- tiktoken-rs batch avg: {:#?}",
            Duration::from_nanos(avg_batch_time_ns)
        );
        let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
        println!(
            "- tiktoken-rs sample avg: {:#?}",
            Duration::from_nanos(avg_sample_time_ns)
        );
        let b_p_ns = avg_sample_size as f64 / avg_sample_time_ns as f64;
        let b_p_s = b_p_ns * 1e9;
        let mb_p_s = b_p_s / 1e6;
        println!("- tiktoken-rs avg bps: {:.2} MB/s", mb_p_s);
    }

    println!();
    println!("Observed Bytes/Token Stats:");
    println!("- total bytes: {}", total_sample_bytes);
    println!("- total wordchipper tokens: {}", total_wc_token_count);
    println!("- total tiktoken-rs tokens: {}", total_tt_token_count);
    println!(
        "- wordchipper byte/token: {:.2}",
        total_sample_bytes as f64 / total_wc_token_count as f64
    );
    println!(
        "- tiktoken-rs byte/token: {:.2}",
        total_sample_bytes as f64 / total_tt_token_count as f64
    );

    println!();
    let num_batches1 = token_batches.len();
    println!("Timing Decode:");

    let segmentor: TextSegmentor = TextSegmentor::from_config(vocab.segmentation.clone());

    let mut wc_batch_times_ns = vec![];
    let mut tt_batch_times_ns = vec![];
    for (idx, sample) in sample_batches.iter().enumerate() {
        let batch = &token_batches[idx];
        let t0 = std::time::Instant::now();
        let wc_decoded = wc_tokenizer
            .decoder
            .try_decode_batch_to_strings(batch)
            .unwrap();
        let t1 = std::time::Instant::now();
        let delay = t1.duration_since(t0);
        wc_batch_times_ns.push(delay.as_nanos() as u64);

        let expected = sample
            .iter()
            .map(|s| String::from_utf8_lossy(segmentor.rewrite(s).as_bytes()).to_string())
            .collect::<Vec<_>>();

        for (s, d) in expected.iter().zip(wc_decoded.iter()) {
            if s != d {
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

        {
            use rayon::prelude::*;

            let t0 = std::time::Instant::now();
            let tt_decoded = batch
                .iter()
                .map(|tokens| tt_bpe.decode(tokens.clone()).unwrap())
                .collect::<Vec<_>>();
            let t1 = std::time::Instant::now();
            let delay = t1.duration_since(t0);
            tt_batch_times_ns.push(delay.as_nanos() as u64);

            for (s, d) in expected.iter().zip(tt_decoded.iter()) {
                if s != d {
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
        }
    }

    {
        let avg_batch_time_ns = wc_batch_times_ns.iter().sum::<u64>() / num_batches1 as u64;
        println!(
            "- wordchipper batch avg: {:?}",
            Duration::from_nanos(avg_batch_time_ns)
        );

        let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
        println!(
            "- wordchipper sample avg: {:?}",
            Duration::from_nanos(avg_sample_time_ns)
        );
    }
    {
        let avg_batch_time_ns = tt_batch_times_ns.iter().sum::<u64>() / num_batches1 as u64;
        println!(
            "- tiktoken-rs batch avg: {:?}",
            Duration::from_nanos(avg_batch_time_ns)
        );

        let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
        println!(
            "- tiktoken-rs sample avg: {:?}",
            Duration::from_nanos(avg_sample_time_ns)
        );
    }
    Ok(())
}

/*
pub fn batch_score(
    actual: &[String],
    expected: &[String],
) -> f64 {
    score_batch(actual, expected).iter().sum::<f64>() / actual.len() as f64
}

pub fn score_batch(
    actual: &[String],
    expected: &[String],
) -> Vec<f64> {
    use rayon::prelude::*;
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected.iter())
        .collect::<Vec<_>>()
        .par_iter()
        .map(|(a, e)| edit_score(a, e))
        .collect::<Vec<_>>()
}

pub fn edit_score(
    actual: &str,
    expected: &str,
) -> f64 {
    let distance = edit_distance(actual, expected);
    let size = expected.len();

    (size as isize - distance as isize).abs() as f64 / (size as f64)
}
*/
