//! # Vocab Factory Support

#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[cfg(feature = "std")]
use crate::resources::ResourceLoader;
#[cfg(feature = "std")]
use crate::vocab::UnifiedTokenVocab;
use crate::{
    alloc::{string::String, vec::Vec},
    regex::{ConstRegexWrapperPattern, RegexWrapperPattern},
    resources::ConstKeyedResource,
    spanning::TextSpanningConfig,
    types::TokenType,
};

/// A trait for pretrained tokenizer factories.
pub trait VocabFactory {
    /// Get the regex pattern for this tokenizer.
    fn pattern(&self) -> RegexWrapperPattern;

    /// List the special tokens for this tokenizer.
    fn special_tokens<T: TokenType>(&self) -> Vec<(String, T)>;

    /// Get the spanning config for this tokenizer.
    fn spanning_config<T: TokenType>(&self) -> TextSpanningConfig<T>;

    /// Load the pretrained vocabulary through the loader.
    #[cfg(feature = "std")]
    fn load_vocab<T: TokenType, L: ResourceLoader>(
        &self,
        loader: &mut L,
    ) -> anyhow::Result<UnifiedTokenVocab<T>>;
}

/// A pretrained tokenizer bundle.
pub struct ConstBase64VocabFactory {
    /// The name of the tokenizer.
    pub name: &'static str,

    /// A (key, resource) pair.
    pub resource: ConstKeyedResource,

    /// The tokenizer regex pattern.
    pub pattern: ConstRegexWrapperPattern,

    /// A generator for special tokens.
    pub special_builder: &'static dyn Fn() -> Vec<(String, usize)>,
}

impl VocabFactory for ConstBase64VocabFactory {
    /// Get the regex pattern for this tokenizer.
    fn pattern(&self) -> RegexWrapperPattern {
        self.pattern.to_pattern()
    }

    /// List the special tokens for this tokenizer.
    fn special_tokens<T: TokenType>(&self) -> Vec<(String, T)> {
        (self.special_builder)()
            .into_iter()
            .map(|(s, t)| (s, T::from_usize(t).unwrap()))
            .collect()
    }

    /// Load the spanning config for this tokenizer.
    fn spanning_config<T: TokenType>(&self) -> TextSpanningConfig<T> {
        TextSpanningConfig::from_pattern(self.pattern()).with_special_words(self.special_tokens())
    }

    /// Load the pretrained vocabulary through the loader.
    #[cfg(feature = "std")]
    fn load_vocab<T: TokenType, L: ResourceLoader>(
        &self,
        loader: &mut L,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        let path = loader.load_resource_path(self.resource.clone())?;
        self.load_vocab_path(path)
    }
}

impl ConstBase64VocabFactory {
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
