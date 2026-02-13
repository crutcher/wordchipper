//! # `OpenAI` Pretrained Vocabulary Loaders

use std::{io::BufRead, path::Path};

#[cfg(feature = "download")]
use crate::disk_cache::WordchipperDiskCache;
use crate::{
    pretrained::openai::{
        OA_CL100K_BASE_PATTERN,
        OA_O200K_BASE_PATTERN,
        OA_P50K_BASE_PATTERN,
        OA_R50K_BASE_PATTERN,
        resources::{
            OA_CL100K_BASE_TIKTOKEN_RESOURCE,
            OA_O200K_BASE_TIKTOKEN_RESOURCE,
            OA_P50K_BASE_TIKTOKEN_RESOURCE,
            OA_R50K_BASE_TIKTOKEN_RESOURCE,
        },
        specials::{
            oa_o200k_harmony_special_tokens,
            oa_p50k_base_special_tokens,
            oa_p50k_edit_special_tokens,
            oa_r50k_base_special_tokens,
        },
    },
    regex::RegexWrapperPattern,
    spanning::TextSpanningConfig,
    types::TokenType,
    vocab::{
        UnifiedTokenVocab,
        utility::{ConstKeyedResource, ConstVocabularyFactory},
    },
};

/// `OpenAI` Pretrained Tokenizer types.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    strum_macros::EnumString,
    strum_macros::EnumIter,
    strum_macros::Display,
)]
#[non_exhaustive]
pub enum OATokenizer {
    /// GPT-2 "`r50k_base`" tokenizer.
    #[strum(serialize = "r50k_base")]
    R50kBase,

    /// GPT-2 "`p50k_base`" tokenizer.
    #[strum(serialize = "p50k_base")]
    P50kBase,

    /// GPT-2 "`p50k_edit`" tokenizer.
    #[strum(serialize = "p50k_edit")]
    P50kEdit,

    /// GPT-3 "`cl100k_base`" tokenizer.
    #[strum(serialize = "cl100k_base")]
    Cl100kBase,

    /// GPT-5 "`o200k_base`" tokenizer.
    #[strum(serialize = "o200k_base")]
    O200kBase,

    /// GPT-5 "`o200k_harmony`" tokenizer.
    #[strum(serialize = "o200k_harmony")]
    O200kHarmony,
}

impl OATokenizer {
    /// Get the tokenizer vocabulary factory.
    pub fn factory(&self) -> &ConstVocabularyFactory {
        use OATokenizer::*;
        match self {
            R50kBase => &OA_R50K_BASE_VOCAB_FACTORY,
            P50kBase => &OA_P50K_BASE_VOCAB_FACTORY,
            P50kEdit => &OA_P50K_EDIT_VOCAB_FACTORY,
            Cl100kBase => &OA_CL100K_BASE_VOCAB_FACTORY,
            O200kBase => &OA_O200K_BASE_VOCAB_FACTORY,
            O200kHarmony => &OA_O200K_HARMONY_VOCAB_FACTORY,
        }
    }

    /// Get the tokenizer regex pattern.
    pub fn pattern(&self) -> RegexWrapperPattern {
        self.factory().pattern()
    }

    /// Get the tokenizer special tokens.
    pub fn special_tokens<T: TokenType>(&self) -> Vec<(String, T)> {
        self.factory().special_tokens()
    }

    /// Get the tokenizer spanning config.
    pub fn spanning_config<T: TokenType>(&self) -> TextSpanningConfig<T> {
        self.factory().spanning_config()
    }

    /// Load pretrained `OpenAI` tokenizer vocabulary.
    ///
    /// Downloads and caches resources using the `disk_cache`.
    #[cfg(feature = "download")]
    pub fn load_vocab<T: TokenType>(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        self.factory().load_vocab(disk_cache)
    }

    /// Load pretrained `OpenAI` tokenizer vocabulary from disk.
    pub fn load_path<T: TokenType>(
        &self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        self.factory().load_vocab_path(path)
    }

    /// Read pretrained `OpenAI` tokenizer vocabulary from a reader.
    pub fn read_vocab<T: TokenType, R: BufRead>(
        &self,
        reader: R,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        self.factory().read_vocab(reader)
    }
}

/// Shared download context key.
const OA_KEY: &str = "openai";

/// The "`r50k_base`" tokenizer.
pub const OA_R50K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "r50k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "r50k_base"],
        resource: OA_R50K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_R50K_BASE_PATTERN,
    special_builder: &oa_r50k_base_special_tokens,
};

/// The "`p50k_base`" tokenizer.
pub const OA_P50K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "p50k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "p50k_base"],
        resource: OA_P50K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_P50K_BASE_PATTERN,
    special_builder: &oa_p50k_base_special_tokens,
};

/// The "`p50k_base`" tokenizer.
pub const OA_P50K_EDIT_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "p50k_edit",
    resource: OA_P50K_BASE_VOCAB_FACTORY.resource,
    pattern: OA_P50K_BASE_VOCAB_FACTORY.pattern,
    special_builder: &oa_p50k_edit_special_tokens,
};

/// The "`cl100k_base`" tokenizer.
pub const OA_CL100K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "cl100k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "cl100k_base"],
        resource: OA_CL100K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_CL100K_BASE_PATTERN,
    special_builder: &oa_p50k_edit_special_tokens,
};

/// The "`o200k_base`" tokenizer.
pub const OA_O200K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "o200k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "o200k_base"],
        resource: OA_O200K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_O200K_BASE_PATTERN,
    special_builder: &oa_p50k_edit_special_tokens,
};

/// The "`o200k_harmony`" tokenizer.
pub const OA_O200K_HARMONY_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "o200k_harmony",
    resource: OA_O200K_BASE_VOCAB_FACTORY.resource,
    pattern: OA_O200K_BASE_VOCAB_FACTORY.pattern,
    special_builder: &oa_o200k_harmony_special_tokens,
};

#[cfg(test)]
mod test {
    use core::str::FromStr;

    use super::*;

    #[test]
    fn test_oa_tokenizer() {
        assert_eq!(OATokenizer::R50kBase.to_string(), "r50k_base");
        assert_eq!(OATokenizer::P50kBase.to_string(), "p50k_base");
        assert_eq!(OATokenizer::P50kEdit.to_string(), "p50k_edit");
        assert_eq!(OATokenizer::Cl100kBase.to_string(), "cl100k_base");
        assert_eq!(OATokenizer::O200kBase.to_string(), "o200k_base");
        assert_eq!(OATokenizer::O200kHarmony.to_string(), "o200k_harmony");

        assert_eq!(
            OATokenizer::from_str("r50k_base").unwrap(),
            OATokenizer::R50kBase
        );
        assert_eq!(
            OATokenizer::from_str("p50k_base").unwrap(),
            OATokenizer::P50kBase
        );
        assert_eq!(
            OATokenizer::from_str("p50k_edit").unwrap(),
            OATokenizer::P50kEdit
        );
        assert_eq!(
            OATokenizer::from_str("cl100k_base").unwrap(),
            OATokenizer::Cl100kBase
        );
        assert_eq!(
            OATokenizer::from_str("o200k_base").unwrap(),
            OATokenizer::O200kBase
        );
        assert_eq!(
            OATokenizer::from_str("o200k_harmony").unwrap(),
            OATokenizer::O200kHarmony
        );
    }
}
