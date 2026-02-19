use std::{collections::HashSet, time::Duration};

use arrow::array::{Array, StringArray};
use clap::Parser;
use similar::{ChangeTag, TextDiff};
use wordchipper::{
    UnifiedTokenVocab, VocabIndex,
    compat::slices::{inner_slice_view, inner_str_view},
    pretrained::openai::OA_O200K_BASE_PATTERN,
    training::BinaryPairVocabTrainerOptions,
    vocab::{ByteMapVocab, io::save_base64_span_map_path},
};
use wordchipper_data::dataset::DatasetCacheConfig;

/// Example encoders trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Shards to load.
    #[arg(long, num_args = 1.., default_values_t = vec![0,1,2,3,4,5,6,7])]
    pub shards: Vec<usize>,

    /// Vocab size.
    #[arg(long, default_value = "199997")]
    pub vocab_size: usize,

    /// Time the avg encode/decode.
    #[arg(long, default_value = "false")]
    pub time_encode_decode: bool,

    /// Encode/Decode Batch size.
    #[arg(long, default_value = "512")]
    pub batch_size: usize,

    /// Optional Tiktoken save path.
    #[arg(long)]
    pub tiktoken_save_path: Option<String>,

    /// Number of timing batches to use.
    #[arg(long, default_value = "20")]
    pub num_timing_batches: usize,

    /// Enable verbose output.
    #[arg(long, default_value = "false")]
    pub verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.verbose {
        println!("{:#?}", args);
    }

    let cache_config = DatasetCacheConfig::default().with_cache_dir(args.dataset_dir);
    if args.verbose {
        println!("{:#?}", cache_config);
    }

    let shards: Vec<usize> = {
        let max_shard = cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for &idx in &args.shards {
            assert!(idx < max_shard, "shard index out of range");
            collected.insert(idx);
        }
        let mut shards: Vec<usize> = collected.into_iter().collect();
        shards.sort();
        shards
    };

    let mut cache = cache_config.init()?;

    // TODO: `indicatif` for optional progress bar for users waiting on this.
    println!("Loading Shards: {shards:?}");
    println!("...");
    cache.load_shards(&shards)?;

    type T = u32;
    type K = String;
    type C = u32;

    println!();
    println!("Training Tokenizer on shards: {:?}", shards);
    let t0 = std::time::Instant::now();

    let vocab_size = args.vocab_size;
    let options = BinaryPairVocabTrainerOptions::new(OA_O200K_BASE_PATTERN, vocab_size);

    let mut trainer = options.init::<K, C>();

    for &shard in &shards {
        println!("- shard: {}", shard);
        for batch in cache.read_cached_batches(shard)? {
            let batch = batch?;

            let samples = batch
                .column_by_name("text")
                .expect("failed to find 'text' column in batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .filter_map(|s| s.map(|s| s.to_string()));

            trainer.update_from_samples(samples);
        }
    }

    let byte_vocab: ByteMapVocab<T> = Default::default();

    println!("- train");
    let vocab: UnifiedTokenVocab<T> = trainer.train(byte_vocab.clone()).expect("training failed");

    let training_duration = std::time::Instant::now().duration_since(t0);
    println!("- training_duration: {:.2?}", training_duration);
    println!("- vocab_size: {:?}", vocab.max_token());

    if let Some(path) = args.tiktoken_save_path {
        save_base64_span_map_path(vocab.span_vocab().span_map(), &path)?;
        println!("- tiktoken vocab: {path:?}");
    }

    if args.time_encode_decode {
        let encoder = vocab.to_default_encoder();
        let decoder = vocab.to_default_decoder();

        let mut samples = Vec::new();
        {
            for batch in cache
                .read_cached_batches(shards[0])?
                .take(args.num_timing_batches)
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

        let sample_batches: Vec<&[String]> = samples.chunks(args.batch_size).collect::<Vec<_>>();
        let num_batches = sample_batches.len();

        println!();
        println!("Timing Config:");
        println!("- batch size: {}", args.batch_size);

        println!();
        println!("Timing Encode:");
        let mut token_batches: Vec<Vec<Vec<T>>> = Vec::with_capacity(sample_batches.len());
        let mut total_token_count = 0;
        let batch_times_ns = sample_batches.iter().map(|batch| {
            let t0 = std::time::Instant::now();
            let token_batch: Vec<Vec<T>> =
                encoder.try_encode_batch(&inner_str_view(batch)).unwrap();
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

        let avg_sample_time_ns = avg_batch_time_ns / args.batch_size as u64;
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
        let batch_size = args.batch_size;
        let num_batches1 = token_batches.len();
        println!("Timing Decode:");

        let batch_times_ns =
            sample_batches
                .iter()
                .zip(token_batches.iter())
                .map(|(sample, batch)| {
                    let t0 = std::time::Instant::now();
                    let decoded_sample = decoder
                        .try_decode_batch_to_strings(&inner_slice_view(batch))
                        .unwrap()
                        .unwrap();

                    let t1 = std::time::Instant::now();

                    for (s, d) in sample.iter().zip(decoded_sample.iter()) {
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
    }

    Ok(())
}
