use burn::tensor::{AsIndex, Slice};
use clap::Parser;
use std::collections::HashSet;
use wordchipper_data::dataset::DatasetCacheConfig;

/// Example Nanochat Data Loader.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("{:#?}", args);

    let cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir);
    println!("{:#?}", cache_config);

    let shards: Vec<usize> = {
        let max_shard = cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for slice in &args.shards {
            for idx in slice.into_iter() {
                let shard = idx.expect_elem_index(max_shard);
                collected.insert(shard);
            }
        }
        let mut shards: Vec<usize> = collected.into_iter().collect();
        shards.sort();
        shards
    };

    let mut cache = cache_config.init()?;

    println!("Loading Shards ...: {shards:?}");
    cache.load_shards(&shards)?;

    Ok(())
}
