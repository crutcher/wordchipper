use arrow::array::StringArray;
use arrow::datatypes::ArrowNativeType;
use clap::Parser;
use similar::{ChangeTag, TextDiff};
use std::sync::Arc;
use std::time::Duration;
use wordchipper::decoders::{DictionaryDecoder, TokenDecoder};
use wordchipper::encoders::{MergeHeapVocabEncoder, TokenEncoder};
use wordchipper::rayon::{ParallelRayonDecoder, ParallelRayonEncoder};
use wordchipper::regex::{RegexWrapperPattern, regex_pool_supplier};
use wordchipper::segmentation::{SegmentationConfig, TextSegmentor};
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::vocab::io::tiktoken_io::load_tiktoken_vocab_path;
use wordchipper::vocab::public::openai::{
    OA_GPT2_R50K_BASE_TIKTOKEN, OA_GPT2_R50K_WORD_PATTERN, oa_gpt2_r50k_specials,
};
use wordchipper_data::dataset::DatasetCacheConfig;

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
    Load {
        /// Path to tokenizer file.
        #[arg(long)]
        tokenizer_file: String,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.verbose {
        println!("{:#?}", args);
    }

    match &args.command {
        Some(Command::Load { tokenizer_file }) => {
            run_load(&args, tokenizer_file)?;
        }
        None => unreachable!(),
    }

    Ok(())
}

#[allow(unused)]
fn run_load(
    args: &Args,
    tokenizer_file: &String,
) -> anyhow::Result<()> {
    let mut cache = DatasetCacheConfig::new()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    type T = u32;

    let pattern: RegexWrapperPattern = OA_GPT2_R50K_WORD_PATTERN.into();

    let r50k_tiktoken = OA_GPT2_R50K_BASE_TIKTOKEN;
    // If we had a download cache, we'd use OA_GPT_R50K_BASE_TIKTOKEN.url here:
    let span_map = load_tiktoken_vocab_path(tokenizer_file)?;

    let segmentation = SegmentationConfig::<T>::from_pattern(pattern.clone()).with_special_words(
        oa_gpt2_r50k_specials()
            .iter()
            .map(|(s, t)| (s, T::from_usize(*t).unwrap())),
    );

    let vocab: Arc<UnifiedTokenVocab<T>> =
        UnifiedTokenVocab::from_span_vocab(segmentation, span_map.into()).into();

    let encoder: MergeHeapVocabEncoder<T> =
        MergeHeapVocabEncoder::<T>::init_with_factory(vocab.clone(), regex_pool_supplier);
    let encoder = ParallelRayonEncoder::new(encoder);

    let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
    let decoder = ParallelRayonDecoder::new(decoder);

    let shards: Vec<usize> = vec![0];
    let num_timing_batches = 20;
    let batch_size = 512;

    println!("Loading Shards: {shards:?}");
    println!("...");
    cache.load_shards(&shards)?;

    let mut samples = Vec::new();
    {
        for batch in cache
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
    let mut total_token_count = 0;
    let batch_times_ns = sample_batches.iter().map(|batch| {
        let t0 = std::time::Instant::now();
        let token_batch: Vec<Vec<T>> = encoder.encode_batch(batch);
        let t1 = std::time::Instant::now();

        total_token_count += token_batch.iter().map(|tokens| tokens.len()).sum::<usize>();

        token_batches.push(token_batch);

        let delay = t1.duration_since(t0);
        delay.as_nanos() as u64
    });

    let avg_batch_time_ns = batch_times_ns.sum::<u64>() / num_batches as u64;
    println!(
        "- batch avg: {:#?}",
        Duration::from_nanos(avg_batch_time_ns)
    );

    let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
    println!(
        "- sample avg: {:#?}",
        Duration::from_nanos(avg_sample_time_ns)
    );
    let b_p_ns = avg_sample_size as f64 / avg_sample_time_ns as f64;
    let b_p_s = b_p_ns * 1e9;
    let mb_p_s = b_p_s / 1e6;
    println!("- avg bps: {:.2} MB/s", mb_p_s);

    println!();
    println!("Observed Bytes/Token Stats:");
    println!("- total bytes: {}", total_sample_bytes);
    println!("- total tokens: {}", total_token_count);
    println!(
        "- sample byte/token: {:.2}",
        total_sample_bytes as f64 / total_token_count as f64
    );

    println!();
    let num_batches1 = token_batches.len();
    println!("Timing Decode:");

    let segmentor: TextSegmentor =
        TextSegmentor::from_config(vocab.segmentation.clone(), regex_pool_supplier);

    let batch_times_ns = sample_batches
        .iter()
        .zip(token_batches.iter())
        .map(|(sample, batch)| {
            let t0 = std::time::Instant::now();
            let decoded_sample = decoder.try_decode_batch_to_strings(batch).unwrap();
            let t1 = std::time::Instant::now();

            let expected = sample
                .iter()
                .map(|s| String::from_utf8_lossy(segmentor.rewrite(s).as_bytes()).to_string())
                .collect::<Vec<_>>();

            for (s, d) in expected.iter().zip(decoded_sample.iter()) {
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

            let delay = t1.duration_since(t0);
            delay.as_nanos() as u64
        });

    let avg_batch_time_ns = batch_times_ns.sum::<u64>() / num_batches1 as u64;
    println!("- batch avg: {:?}", Duration::from_nanos(avg_batch_time_ns));

    let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
    println!(
        "- sample avg: {:?}",
        Duration::from_nanos(avg_sample_time_ns)
    );
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
