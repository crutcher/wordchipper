extern crate core;

use std::{collections::HashMap, io, io::IsTerminal, iter::Iterator, sync::Arc, time::Duration};

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
    spanning::TextSpanner,
    types::TokenType,
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
#[allow(clippy::vec_init_then_push)]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let display_progress = io::stdout().is_terminal();

    let mut shard_data_cache = DatasetCacheConfig::default()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    println!("Model: \"oa:{}\"", args.model);
    println!("- shards: {:?}", args.shards);
    println!("- batch_size: {}", args.batch_size);

    let vocab: UnifiedTokenVocab<Rank> = {
        let mut disk_cache = WordchipperDiskCache::default();
        args.model.load(&mut disk_cache)?
    };
    let spanner = TextSpanner::from_config(vocab.spanning().clone(), None);

    // TODO: complete batch-observer inversion of control for additional tokenizer wrappers.

    let mut candidates: Vec<Arc<dyn TokenizerWrapper<Rank>>> = Vec::new();

    candidates.push(Arc::new(TiktokenWrapper::new(load_tiktoken(args.model)?)));

    candidates.push(Arc::new(WordchipperWrapper::<Rank>::new(
        "wordchipper".to_string(),
        Arc::new(ParallelRayonEncoder::new(Arc::new(
            DefaultTokenEncoder::new(vocab.clone(), None),
        ))),
        Arc::new(ParallelRayonDecoder::new(Arc::new(
            DefaultTokenDecoder::from_unified_vocab(vocab.clone()),
        ))),
    )));

    let mut stats = Vec::new();

    for_each_batch(
        display_progress,
        &args.shards,
        args.batch_size,
        &mut shard_data_cache,
        &mut |str_batch: &[&str]| -> anyhow::Result<()> {
            let mut timings: HashMap<String, BatchTimings> = Default::default();

            for w in candidates.iter() {
                timings.insert(w.name().to_string(), Default::default());
            }

            let mut reference = None;
            for w in candidates.iter() {
                let name = w.name();
                let (w_dur, w_tokens) = timeit(|| w.expect_encode_batch(str_batch));

                match &reference {
                    None => {
                        reference = Some((name, w_tokens));
                    }
                    Some((expected_name, expected_tokens)) => {
                        verify_encode(str_batch, name, &w_tokens, expected_name, expected_tokens)?;
                    }
                }

                timings.get_mut(name).unwrap().encode = w_dur;
            }

            let batch_tokens = reference.unwrap().1;
            let batch_view: Vec<&[Rank]> = inner_slice_view(&batch_tokens);

            for w in candidates.iter() {
                let name = w.name();
                let (w_dur, w_batch) = timeit(|| w.expect_decode_batch(&batch_view));

                verify_decode(&batch_view, str_batch, name, &w_batch);

                timings.get_mut(name).unwrap().decode = w_dur;
            }

            let sample_bytes = str_batch.iter().map(|s| s.len()).collect();
            let token_counts = batch_view.iter().map(|t| t.len()).collect();

            stats.push(BatchStats {
                sample_bytes,
                token_counts,
                timings,
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
    for w in candidates.iter() {
        let name = w.name();
        let total_duration = stats
            .iter()
            .map(|s| s.timings[name].encode)
            .sum::<Duration>();
        let mean_duration = total_duration / num_batches as u32;
        print_timing(name, mean_duration, avg_batch_bytes, args.batch_size);
    }

    println!();
    println!("Decoder Times:");
    for w in candidates.iter() {
        let name = w.name();
        let total_duration = stats
            .iter()
            .map(|s| s.timings[name].decode)
            .sum::<Duration>();
        let mean_duration = total_duration / num_batches as u32;
        print_timing(name, mean_duration, avg_batch_bytes, args.batch_size);
    }

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
    actual_name: &str,
    actual_batch: &[Vec<Rank>],
    expected_name: &str,
    expected_batch: &[Vec<Rank>],
) -> anyhow::Result<()> {
    assert_eq!(source_batch.len(), actual_batch.len());
    assert_eq!(source_batch.len(), expected_batch.len());
    for (i, s) in source_batch.iter().enumerate() {
        let actual_tokens = &actual_batch[i];
        let expected_tokens = &expected_batch[i];

        if actual_tokens == expected_tokens {
            continue;
        }

        bail!(
            "ENCODER MISMATCH:\n- source: {}\n- {actual_name}: {:?}\n- {expected_name}: {:?}",
            s,
            actual_tokens,
            expected_tokens,
        )
    }
    Ok(())
}

pub fn verify_decode(
    batch_tokens: &[&[Rank]],
    expected_batch: &[&str],
    actual_name: &str,
    actual_batch: &[String],
) -> anyhow::Result<()> {
    assert_eq!(batch_tokens.len(), expected_batch.len());
    assert_eq!(batch_tokens.len(), actual_batch.len());

    for (i, &expected_str) in expected_batch.iter().enumerate() {
        let actual_str = &actual_batch[i];

        if actual_str == expected_str {
            let tokens = batch_tokens[i];

            bail!(
                "DECODER MISMATCH:\n- tokens: {:?}\n- expected: {}\n- {}: {}",
                tokens,
                expected_str,
                actual_name,
                actual_str,
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

pub trait TokenizerWrapper<T: TokenType> {
    fn name(&self) -> &str;

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>>;

    fn expect_encode_batch(
        &self,
        batch: &[&str],
    ) -> Vec<Vec<T>> {
        self.encode_batch(batch)
            .unwrap_or_else(|_| panic!("failed to encode batch with \"{}\"", self.name()))
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<Vec<String>>;

    fn expect_decode_batch(
        &self,
        batch: &[&[T]],
    ) -> Vec<String> {
        self.decode_batch(batch)
            .unwrap_or_else(|_| panic!("failed to decode batch with \"{}\"", self.name()))
    }
}

pub struct TiktokenWrapper {
    inner: CoreBPE,
}

impl TiktokenWrapper {
    pub fn new(inner: CoreBPE) -> Self {
        Self { inner }
    }
}

impl TokenizerWrapper<Rank> for TiktokenWrapper {
    fn name(&self) -> &str {
        "tiktoken"
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<Rank>>> {
        Ok(batch
            .par_iter()
            .map(|s| self.inner.encode_with_special_tokens(s))
            .collect::<Vec<_>>())
    }

    fn decode_batch(
        &self,
        batch: &[&[Rank]],
    ) -> anyhow::Result<Vec<String>> {
        Ok(batch
            .par_iter()
            .map(|tokens| self.inner.decode(tokens.to_vec()).unwrap())
            .collect::<Vec<_>>())
    }
}

pub struct WordchipperWrapper<T: TokenType> {
    name: String,
    encoder: Arc<dyn TokenEncoder<T>>,
    decoder: Arc<dyn TokenDecoder<T>>,
}

impl<T: TokenType> WordchipperWrapper<T> {
    pub fn new(
        name: String,
        encoder: Arc<dyn TokenEncoder<T>>,
        decoder: Arc<dyn TokenDecoder<T>>,
    ) -> Self {
        Self {
            name,
            encoder,
            decoder,
        }
    }
}

impl<T: TokenType> TokenizerWrapper<T> for WordchipperWrapper<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>> {
        self.encoder.try_encode_batch(batch)
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<Vec<String>> {
        Ok(self.decoder.try_decode_batch_to_strings(batch)?.unwrap())
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct BatchTimings {
    pub encode: Duration,
    pub decode: Duration,
}

struct BatchStats {
    pub sample_bytes: Vec<usize>,
    pub token_counts: Vec<usize>,

    pub timings: HashMap<String, BatchTimings>,
}

impl BatchStats {
    pub fn len(&self) -> usize {
        self.sample_bytes.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.token_counts.iter().sum::<usize>()
    }

    pub fn batch_bytes(&self) -> usize {
        self.sample_bytes.iter().sum::<usize>()
    }

    pub fn avg_sample_bytes(&self) -> usize {
        self.batch_bytes() / self.len()
    }
}
