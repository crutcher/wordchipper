use std::{collections::HashSet, sync::Arc};

use arrow::array::{Array, StringArray};
use clap::Parser;
use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    pretrained::openai::OA_R50K_BASE_PATTERN,
    vocab::io::save_base64_span_map_path,
};
use wordchipper_data::dataset::DatasetCacheConfig;
use wordchipper_training::{BPETRainerOptions, BPETrainer};

/// Example encoders trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Shards to load.
    #[arg(long, num_args = 1.., default_values_t = vec![0,1,2,3,4,5,6,7])]
    pub nanochat_shards: Vec<usize>,

    /// Vocab size.
    #[arg(long, default_value = "199997")]
    pub vocab_size: usize,

    /// Encode/Decode Batch size.
    #[arg(long, default_value = "512")]
    pub batch_size: usize,

    /// Regex pattern for tokenizer.
    #[arg(long, default_value_t = OA_R50K_BASE_PATTERN.as_str().to_string())]
    pub regex: String,

    /// Vocab save path (.tiktoken format).
    #[arg(long)]
    pub vocab_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:#?}", args);

    let cache_config = DatasetCacheConfig::default().with_cache_dir(args.dataset_dir);
    println!("{:#?}", cache_config);

    let shards: Vec<usize> = {
        let max_shard = cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for &idx in &args.nanochat_shards {
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

    println!();
    println!("Training Tokenizer on shards: {:?}", shards);
    let t0 = std::time::Instant::now();

    let vocab_size = args.vocab_size;
    let options = BPETRainerOptions::new(args.regex, vocab_size);

    let mut trainer: BPETrainer = options.init();

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

    println!("- train");
    let vocab: Arc<UnifiedTokenVocab<u32>> = trainer
        .train(Default::default())
        .expect("training failed")
        .into();

    let training_duration = std::time::Instant::now().duration_since(t0);
    println!("- training_duration: {:.2?}", training_duration);
    println!("- vocab_size: {:?}", vocab.max_token());

    save_base64_span_map_path(vocab.span_vocab().span_map(), &args.vocab_path)?;
    println!("- tiktoken vocab: {:?}", args.vocab_path);

    Ok(())
}
