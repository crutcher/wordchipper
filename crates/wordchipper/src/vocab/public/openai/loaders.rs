//! # `OpenAI` Pretrained Vocabulary Loaders

use crate::disk_cache::WordchipperDiskCache;
use crate::regex::RegexWrapperPattern;
use crate::spanner::TextSpanConfig;
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;
use crate::vocab::io::load_tiktoken_vocab_path;
use crate::vocab::public::openai::patterns::{
    OA_GPT2_R50K_WORD_PATTERN, OA_GPT3_CL100K_WORD_PATTERN,
};
use crate::vocab::public::openai::resources::{
    OA_GPT2_P50K_BASE_TIKTOKEN, OA_GPT2_R50K_BASE_TIKTOKEN, OA_GPT3_CL100K_BASE_TIKTOKEN,
    OA_GPT5_O200K_BASE_TIKTOKEN,
};
use crate::vocab::public::openai::specials::{
    oa_gpt2_p50k_base_specials, oa_gpt2_p50k_edit_specials, oa_gpt2_r50k_specials,
    oa_gpt3_cl100k_edit_specials, oa_gpt5_o200k_harmony_specials, oa_gt5_o200k_base_specials,
};
use crate::vocab::utility::resource_tools::ConstUrlResource;
use strum_macros::EnumString;

/// Shared download context key.
const OA_KEY: &str = "openai";

/// `OpenAI` Pretrained Tokenizer types.
#[derive(Clone, Debug, PartialEq, EnumString, strum_macros::Display)]
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
    /// Load a pretrained `OpenAI` tokenizer vocabulary.
    pub fn load<T: TokenType>(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        match self {
            OATokenizer::R50kBase => load_r50k_base_vocab::<T>(disk_cache),
            OATokenizer::P50kBase => load_p50k_base_vocab::<T>(disk_cache),
            OATokenizer::P50kEdit => load_p50k_edit_vocab::<T>(disk_cache),
            OATokenizer::Cl100kBase => load_cl100k_base_vocab::<T>(disk_cache),
            OATokenizer::O200kBase => load_o200k_base_vocab::<T>(disk_cache),
            OATokenizer::O200kHarmony => load_o200k_harmony_vocab::<T>(disk_cache),
        }
    }
}

fn load_common_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache,
    context: &[&str],
    vocab_resource: &ConstUrlResource,
    pattern: RegexWrapperPattern,
    special_tokens: &[(String, usize)],
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    let span_map = load_tiktoken_vocab_path(disk_cache.load_cached_path(
        context,
        vocab_resource.urls,
        true,
    )?)?;

    let segmentation = TextSpanConfig::<T>::from_pattern(pattern.clone()).with_special_words(
        special_tokens
            .iter()
            .map(|(s, t)| (s, T::from_usize(*t).unwrap())),
    );

    let vocab = UnifiedTokenVocab::from_span_vocab(segmentation, span_map.into());
    Ok(vocab)
}

/// Load GPT-2 "`r50k`" pretrained vocabulary.
pub fn load_r50k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab(
        disk_cache,
        &[OA_KEY, "r50k"],
        &OA_GPT2_R50K_BASE_TIKTOKEN,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt2_r50k_specials().to_vec(),
    )
}

/// Load GPT-2 "`p50k_base`" pretrained vocabulary.
pub fn load_p50k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab(
        disk_cache,
        &[OA_KEY, "p50k"],
        &OA_GPT2_P50K_BASE_TIKTOKEN,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt2_p50k_base_specials().to_vec(),
    )
}

/// Load GPT-2 "`p50k_edit`" pretrained vocabulary.
pub fn load_p50k_edit_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab(
        disk_cache,
        &[OA_KEY, "p50k"],
        &OA_GPT2_P50K_BASE_TIKTOKEN,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt2_p50k_edit_specials().to_vec(),
    )
}

/// Load GPT-3 "`cl100k`" pretrained vocabulary.
pub fn load_cl100k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab(
        disk_cache,
        &[OA_KEY, "cl100k"],
        &OA_GPT3_CL100K_BASE_TIKTOKEN,
        OA_GPT3_CL100K_WORD_PATTERN.into(),
        &oa_gpt3_cl100k_edit_specials().to_vec(),
    )
}

/// Load GPT-5 "`o200k_base`" pretrained vocabulary.
pub fn load_o200k_base_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab(
        disk_cache,
        &[OA_KEY, "o200k"],
        &OA_GPT5_O200K_BASE_TIKTOKEN,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gt5_o200k_base_specials().to_vec(),
    )
}

/// Load GPT-5 "`o200k_harmony`" pretrained vocabulary.
pub fn load_o200k_harmony_vocab<T: TokenType>(
    disk_cache: &mut WordchipperDiskCache
) -> anyhow::Result<UnifiedTokenVocab<T>> {
    load_common_vocab(
        disk_cache,
        &[OA_KEY, "o200k"],
        &OA_GPT5_O200K_BASE_TIKTOKEN,
        OA_GPT2_R50K_WORD_PATTERN.into(),
        &oa_gpt5_o200k_harmony_specials().to_vec(),
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use core::str::FromStr;

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
