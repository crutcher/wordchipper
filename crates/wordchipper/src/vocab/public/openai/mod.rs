//! Public `OpenAI` Patterns, Constants, and Models.

pub mod patterns;
pub mod resources;
pub mod specials;

#[doc(inline)]
pub use patterns::*;
#[doc(inline)]
pub use resources::*;
#[doc(inline)]
pub use specials::*;

#[cfg(feature = "download")]
mod pretrained {
    use crate::regex::RegexWrapperPattern;
    use crate::segmentation::SegmentationConfig;
    use crate::types::TokenType;
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::io::load_tiktoken_vocab_path;
    use crate::vocab::public::openai::{
        OA_GPT2_R50K_BASE_TIKTOKEN, OA_GPT2_R50K_WORD_PATTERN, oa_gpt2_r50k_specials,
    };
    use wordchipper_disk_cache::WordchipperDiskCache;

    /// Load gpt2-r50k pretrained vocabulary.
    pub fn experimental_load_gpt2_r50k<T: TokenType>(
        disk_cache: &mut WordchipperDiskCache
    ) -> anyhow::Result<UnifiedTokenVocab<T>> {
        let span_map = load_tiktoken_vocab_path(disk_cache.load_cached_path(
            &["openai", "gpt2-r50k"],
            OA_GPT2_R50K_BASE_TIKTOKEN.urls,
            true,
        )?)?;

        let pattern: RegexWrapperPattern = OA_GPT2_R50K_WORD_PATTERN.into();

        let segmentation = SegmentationConfig::<T>::from_pattern(pattern.clone())
            .with_special_words(
                oa_gpt2_r50k_specials()
                    .iter()
                    .map(|(s, t)| (s, T::from_usize(*t).unwrap())),
            );

        let vocab = UnifiedTokenVocab::from_span_vocab(segmentation, span_map.into());
        Ok(vocab)
    }
}
#[cfg(feature = "download")]
#[doc(inline)]
pub use pretrained::*;
