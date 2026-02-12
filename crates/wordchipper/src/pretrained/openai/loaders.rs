//! # `OpenAI` Pretrained Vocabulary Loaders

use std::path::{Path, PathBuf};

#[cfg(feature = "download")]
use crate::disk_cache::WordchipperDiskCache;
use crate::{
    pretrained::openai::{
        patterns::{
            OA_GPT2_R50K_WORD_PATTERN,
            OA_GPT3_CL100K_WORD_PATTERN,
            OA_GPT5_O220K_WORD_PATTERN,
        },
        resources::{
            OA_GPT2_P50K_BASE_TIKTOKEN,
            OA_GPT2_R50K_BASE_TIKTOKEN,
            OA_GPT3_CL100K_BASE_TIKTOKEN,
            OA_GPT5_O200K_BASE_TIKTOKEN,
        },
        specials::{
            oa_gpt2_p50k_base_specials,
            oa_gpt2_p50k_edit_specials,
            oa_gpt2_r50k_specials,
            oa_gpt3_cl100k_edit_specials,
            oa_gpt5_o200k_harmony_specials,
            oa_gt5_o200k_base_specials,
        },
    },
    regex::RegexWrapperPattern,
    spanning::TextSpanningConfig,
    types::TokenType,
    vocab::{UnifiedTokenVocab, io::load_tiktoken_vocab_path},
};

/// Shared download context key.
const OA_KEY: &str = "openai";

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
    /// Load pretrained `OpenAI` tokenizer vocabulary.
    ///
    /// Downloads and caches resources using the `disk_cache`.
    #[cfg(feature = "download")]
    pub fn load<T: TokenType>(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        use OATokenizer::*;
        match self {
            R50kBase => load_r50k_base_vocab::<T>(disk_cache),
            P50kBase => load_p50k_base_vocab::<T>(disk_cache),
            P50kEdit => load_p50k_edit_vocab::<T>(disk_cache),
            Cl100kBase => load_cl100k_base_vocab::<T>(disk_cache),
            O200kBase => load_o200k_base_vocab::<T>(disk_cache),
            O200kHarmony => load_o200k_harmony_vocab::<T>(disk_cache),
        }
    }

    /// Load pretrained `OpenAI` tokenizer vocabulary from disk.
    pub fn load_path<T: TokenType>(
        &self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        use OATokenizer::*;
        match self {
            R50kBase => load_r50k_base_vocab_path::<T>(path),
            P50kBase => load_p50k_base_vocab_path::<T>(path),
            P50kEdit => load_p50k_edit_vocab_path::<T>(path),
            Cl100kBase => load_cl100k_base_vocab_path::<T>(path),
            O200kBase => load_o200k_base_vocab_path::<T>(path),
            O200kHarmony => load_o200k_harmony_vocab_path::<T>(path),
        }
    }
}

fn load_common_vocab_path<T: TokenType>(
    data_path: impl AsRef<Path>,
    pattern: RegexWrapperPattern,
    special_tokens: &[(String, usize)],
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    let data_path = data_path.as_ref();
    let span_map = load_tiktoken_vocab_path(data_path)?;

    let spanning = TextSpanningConfig::<T>::from_pattern(pattern.clone()).with_special_words(
        special_tokens
            .iter()
            .map(|(s, t)| (s, T::from_usize(*t).unwrap())),
    );

    UnifiedTokenVocab::from_span_vocab(spanning, span_map.into())
}

/// Load GPT-2 "`r50k`" pretrained vocabulary.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn load_r50k_base_vocab<T: TokenType>(
    disk_cache: &mut crate::disk_cache::WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_r50k_base_vocab_path(disk_cache.load_cached_path(
        &[OA_KEY, "r50k"],
        OA_GPT2_R50K_BASE_TIKTOKEN.urls,
        true,
    )?)
}

/// Load GPT-2 "`r50k`" pretrained vocabulary from disk.
pub fn load_r50k_base_vocab_path<T: TokenType>(
    path: impl AsRef<Path>
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab_path(
        path,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt2_r50k_specials(),
    )
}

/// Fetch the GPT-2 "`p50k_base`" pretrained vocabulary data path.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn fetch_p50k_base_data(disk_cache: &mut WordchipperDiskCache) -> anyhow::Result<PathBuf> {
    disk_cache.load_cached_path(&[OA_KEY, "p50k"], OA_GPT2_P50K_BASE_TIKTOKEN.urls, true)
}

/// Load GPT-2 "`p50k_base`" pretrained vocabulary from disk.
pub fn load_p50k_base_vocab_path<T: TokenType>(
    path: impl AsRef<Path>
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab_path(
        path,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt2_p50k_base_specials().to_vec(),
    )
}

/// Load GPT-2 "`p50k_base`" pretrained vocabulary.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn load_p50k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_p50k_base_vocab_path(fetch_p50k_base_data(disk_cache)?)
}

/// Load GPT-2 "`p50k_edit`" pretrained vocabulary.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn load_p50k_edit_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_p50k_edit_vocab_path(disk_cache.load_cached_path(
        &[OA_KEY, "p50k"],
        OA_GPT2_P50K_BASE_TIKTOKEN.urls,
        true,
    )?)
}

/// Load GPT-2 "`p50k_edit`" pretrained vocabulary from disk.
pub fn load_p50k_edit_vocab_path<T: TokenType>(
    path: impl AsRef<Path>
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab_path(
        path,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt2_p50k_edit_specials().to_vec(),
    )
}

/// Load GPT-3 "`cl100k`" pretrained vocabulary.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn load_cl100k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_cl100k_base_vocab_path(disk_cache.load_cached_path(
        &[OA_KEY, "cl100k"],
        OA_GPT3_CL100K_BASE_TIKTOKEN.urls,
        true,
    )?)
}

/// Load GPT-3 "`cl100k`" pretrained vocabulary from disk.
pub fn load_cl100k_base_vocab_path<T: TokenType>(
    path: impl AsRef<Path>
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab_path(
        path,
        OA_GPT3_CL100K_WORD_PATTERN.into(),
        &oa_gpt3_cl100k_edit_specials().to_vec(),
    )
}

/// Load GPT-5 "`o200k_base`" pretrained vocabulary.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn load_o200k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_o200k_base_vocab_path(disk_cache.load_cached_path(
        &[OA_KEY, "o200k"],
        OA_GPT5_O200K_BASE_TIKTOKEN.urls,
        true,
    )?)
}

/// Load GPT-5 "`o200k_base`" pretrained vocabulary.
pub fn load_o200k_base_vocab_path<T: TokenType>(
    path: impl AsRef<Path>
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab_path(
        path,
        OA_GPT5_O220K_WORD_PATTERN.into(),
        &oa_gt5_o200k_base_specials().to_vec(),
    )
}

/// Load GPT-5 "`o200k_harmony`" pretrained vocabulary.
///
/// Downloads and caches resources using the `disk_cache`.
#[cfg(feature = "download")]
pub fn load_o200k_harmony_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_o200k_base_vocab_path(disk_cache.load_cached_path(
        &[OA_KEY, "o200k"],
        OA_GPT5_O200K_BASE_TIKTOKEN.urls,
        true,
    )?)
}

/// Load GPT-5 "`o200k_harmony`" pretrained vocabulary from disk.
pub fn load_o200k_harmony_vocab_path<T: TokenType>(
    path: impl AsRef<Path>
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab_path(
        path,
        OA_GPT5_O220K_WORD_PATTERN.into(),
        &oa_gpt5_o200k_harmony_specials().to_vec(),
    )
}

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
