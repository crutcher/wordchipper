use std::sync::Arc;

use wordchipper::{
    Tokenizer,
    UnifiedTokenVocab,
    disk_cache::{WordchipperDiskCache, WordchipperDiskCacheOptions},
};

use crate::Args;

/// Disk cache control args.
#[derive(clap::Args, Debug)]
pub struct DiskCacheArgs {
    /// Cache directory.
    #[arg(long, default_value = None)]
    cache_dir: Option<String>,
}

/// Setup the disk cache.
pub fn build_disk_cache(args: &Args) -> WordchipperDiskCache {
    let mut options = WordchipperDiskCacheOptions::default();

    if let Some(cache_dir) = &args.disk_cache.cache_dir {
        options = options.with_cache_dir(Some(cache_dir.clone()));
    }

    WordchipperDiskCache::new(options).unwrap()
}

/// The tokenizer mode.
#[derive(Debug, Clone, Copy)]
pub enum TokenizerMode {
    Encode,
    Decode,
}

/// Mode selection for the tokenizer.
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = false)]
pub struct TokenizerModeArgs {
    /// Encode from text to tokens.
    #[arg(long, action=clap::ArgAction::SetTrue)]
    encode: bool,

    /// Decode from tokens to text.
    #[arg(long, action=clap::ArgAction::SetTrue)]
    decode: bool,
}

impl TokenizerModeArgs {
    /// Get the tokenizer mode.
    pub fn mode(&self) -> TokenizerMode {
        if self.encode {
            TokenizerMode::Encode
        } else if self.decode {
            TokenizerMode::Decode
        } else {
            panic!("No tokenizer mode specified.");
        }
    }
}

/// Model selector arg group.
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = false)]
pub struct ModelSelectorArgs {
    /// Model to use for encoding.
    #[arg(long, default_value = "openai::r50k_base")]
    model: String,
}

impl ModelSelectorArgs {
    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Load the vocabulary.
    pub fn load_vocab(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> Result<Arc<UnifiedTokenVocab<u32>>, Box<dyn std::error::Error>> {
        let (_desc, vocab) = wordchipper::load_vocab(self.model(), disk_cache)?;
        Ok(vocab)
    }

    /// Load the tokenizer.
    pub fn load_tokenizer(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> Result<Arc<Tokenizer<u32>>, Box<dyn std::error::Error>> {
        let vocab = self.load_vocab(disk_cache)?;
        let tokenizer = wordchipper::TokenizerOptions::default().build(vocab);
        Ok(tokenizer)
    }
}
