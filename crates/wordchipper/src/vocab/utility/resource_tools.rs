//! # Remote Resource Tools

#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[cfg(feature = "download")]
use crate::disk_cache::WordchipperDiskCache;
#[cfg(feature = "std")]
use crate::vocab::UnifiedTokenVocab;
use crate::{
    alloc::{string::String, vec::Vec},
    regex::{ConstRegexWrapperPattern, RegexWrapperPattern},
    spanning::TextSpanningConfig,
    types::TokenType,
};

/// A resource with a constant URL and optional hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstUrlResource {
    /// The URL associated with this resource.
    pub urls: &'static [&'static str],

    /// The hash associated with this resource, if available.
    pub hash: Option<&'static str>,
}

/// A keyed resource, where the key is a list of strings.
pub struct ConstKeyedResource {
    /// The key associated with this resource.
    ///
    /// This is intended for the [`WordchipperDiskCache`].
    pub key: &'static [&'static str],

    /// The resource associated with this key.
    pub resource: ConstUrlResource,
}

/// A pretrained tokenizer bundle.
pub struct ConstVocabularyFactory {
    /// The name of the tokenizer.
    pub name: &'static str,

    /// A (key, resource) pair.
    pub resource: ConstKeyedResource,

    /// The tokenizer regex pattern.
    pub pattern: ConstRegexWrapperPattern,

    /// A generator for special tokens.
    pub special_builder: &'static dyn Fn() -> Vec<(String, usize)>,
}

impl ConstVocabularyFactory {
    /// Fetch a path to the resource; downloading if necessary.
    #[cfg(feature = "download")]
    pub fn fetch_resource(
        &self,
        disk_cache: &mut WordchipperDiskCache,
        download_if_missing: bool,
    ) -> anyhow::Result<PathBuf> {
        disk_cache.load_cached_path(
            self.resource.key,
            self.resource.resource.urls,
            download_if_missing,
        )
    }

    /// Get the regex pattern for this tokenizer.
    pub fn pattern(&self) -> RegexWrapperPattern {
        self.pattern.to_pattern()
    }

    /// List the special tokens for this tokenizer.
    pub fn special_tokens<T: TokenType>(&self) -> Vec<(String, T)> {
        (self.special_builder)()
            .into_iter()
            .map(|(s, t)| (s, T::from_usize(t).unwrap()))
            .collect()
    }

    /// Load the spanning config for this tokenizer.
    pub fn spanning_config<T: TokenType>(&self) -> TextSpanningConfig<T> {
        TextSpanningConfig::from_pattern(self.pattern()).with_special_words(self.special_tokens())
    }

    /// Load the pretrained vocabulary, downloading if necessary.
    #[cfg(feature = "download")]
    pub fn load_vocab<T: TokenType>(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        let path = self.fetch_resource(disk_cache, true)?;
        self.load_vocab_path(path)
    }

    /// Load the pretrained vocabulary from disk.
    #[cfg(feature = "std")]
    pub fn load_vocab_path<T: TokenType>(
        &self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        let reader = BufReader::new(File::open(path)?);
        self.read_vocab(reader)
    }

    /// Read the pretrained vocabulary from a reader.
    #[cfg(feature = "std")]
    pub fn read_vocab<T: TokenType, R: BufRead>(
        &self,
        reader: R,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        crate::vocab::io::read_base64_unified_vocab(reader, self.spanning_config())
    }
}
