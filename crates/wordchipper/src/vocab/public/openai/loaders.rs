//! # `OpenAI` Pretrained Vocabulary Loaders

use crate::disk_cache::WordchipperDiskCache;
use crate::regex::RegexWrapperPattern;
use crate::segmentation::SegmentationConfig;
use crate::types::TokenType;
use crate::vocab::UnifiedTokenVocab;
use crate::vocab::io::load_tiktoken_vocab_path;
use crate::vocab::public::openai::{
    OA_GPT2_P50K_BASE_TIKTOKEN, OA_GPT2_R50K_BASE_TIKTOKEN, OA_GPT2_R50K_WORD_PATTERN,
    OA_GPT3_CL100K_BASE_TIKTOKEN, OA_GPT3_CL100K_WORD_PATTERN, OA_GPT5_O200K_BASE_TIKTOKEN,
    oa_gpt2_p50k_base_specials, oa_gpt2_p50k_edit_specials, oa_gpt2_r50k_specials,
    oa_gpt3_cl100k_edit_specials, oa_gpt5_o200k_harmony_specials, oa_gt5_o200k_base_specials,
};
use crate::vocab::utility::resource_tools::ConstUrlResource;

/// Shared download context key.
const OA_KEY: &str = "openai";

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

    let segmentation = SegmentationConfig::<T>::from_pattern(pattern.clone()).with_special_words(
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
