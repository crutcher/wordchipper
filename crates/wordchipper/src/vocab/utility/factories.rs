//! # Vocab Factory Support

#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[cfg(feature = "std")]
use crate::resources::ResourceLoader;
#[cfg(feature = "std")]
use crate::vocab::UnifiedTokenVocab;
use crate::{
    alloc::{string::String, sync::Arc, vec::Vec},
    regex::{ConstRegexPattern, RegexPattern},
    resources::ConstKeyedResource,
    spanning::{SpanLexer, TextSpanningConfig},
    types::TokenType,
};

/// A pretrained tokenizer bundle.
pub struct ConstVocabularyFactory {
    /// The name of the tokenizer.
    pub name: &'static str,

    /// A (key, resource) pair.
    pub resource: ConstKeyedResource,

    /// The tokenizer regex pattern.
    pub pattern: ConstRegexPattern,

    /// A generator for special tokens.
    pub special_builder: &'static dyn Fn() -> Vec<(String, usize)>,

    /// Optional factory for the word lexer, overriding the regex-compiled default.
    pub word_lexer_factory: Option<fn() -> Arc<dyn SpanLexer>>,
}

impl ConstVocabularyFactory {
    /// Get the regex pattern for this tokenizer.
    pub fn pattern(&self) -> RegexPattern {
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

    /// Fetch a path to the resource through the loader.
    #[cfg(feature = "std")]
    fn fetch_resource(
        &self,
        loader: &mut dyn ResourceLoader,
    ) -> crate::errors::Result<PathBuf> {
        let res: crate::resources::KeyedResource = self.resource.clone().into();
        loader.load_resource_path(&res)
    }

    /// Load the pretrained vocabulary through the loader.
    #[cfg(feature = "std")]
    pub fn load_vocab<T: TokenType>(
        &self,
        loader: &mut dyn ResourceLoader,
    ) -> crate::errors::Result<UnifiedTokenVocab<T>> {
        let path = self.fetch_resource(loader)?;
        self.load_vocab_path(path)
    }

    /// Load the pretrained vocabulary from disk.
    #[cfg(feature = "std")]
    pub fn load_vocab_path<T: TokenType>(
        &self,
        path: impl AsRef<Path>,
    ) -> crate::errors::Result<UnifiedTokenVocab<T>> {
        let reader = BufReader::new(File::open(path)?);
        self.read_vocab(reader)
    }

    /// Read the pretrained vocabulary from a reader.
    #[cfg(feature = "std")]
    pub fn read_vocab<T: TokenType, R: BufRead>(
        &self,
        reader: R,
    ) -> crate::errors::Result<UnifiedTokenVocab<T>> {
        let mut vocab =
            crate::vocab::io::read_base64_unified_vocab(reader, self.spanning_config())?;
        if let Some(factory) = self.word_lexer_factory {
            vocab.set_word_lexer(factory());
        }
        Ok(vocab)
    }
}
