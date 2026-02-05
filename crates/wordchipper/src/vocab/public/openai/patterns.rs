//! # Patterns

use crate::join_patterns;
use crate::regex::ConstRegexWrapperPattern;

/// The GPT-2 r50k word pattern.
///
/// Faster than [`OA_GPT2_R50K_WORD_PATTERN_SLOW`], optimized for performance.
pub const OA_GPT2_R50K_WORD_PATTERN: ConstRegexWrapperPattern =
    ConstRegexWrapperPattern::Fancy(join_patterns!(
        r"'(?:[sdmt]|ll|ve|re)",
        r" ?\p{L}++",
        r" ?\p{N}++",
        r" ?[^\s\p{L}\p{N}]++",
        r"\s++$",
        r"\s+(?!\S)",
        r"\s",
    ));

/// The original GPT-2 word pattern.
pub const OA_GPT2_R50K_WORD_PATTERN_SLOW: ConstRegexWrapperPattern =
    ConstRegexWrapperPattern::Fancy(join_patterns!(
        r"'s",
        r"'t",
        r"'re",
        r"'ve",
        r"'m",
        r"'ll",
        r"'d",
        r" ?[\p{L}]+",
        r" ?[\p{N}]+",
        r" ?[^\s\p{L}\p{N}]+",
        r"\s+(?!\S)",
        r"\s+",
    ));

/// The GPT-3 cl100K word pattern.
pub const OA_GPT3_CL100K_WORD_PATTERN: ConstRegexWrapperPattern =
    ConstRegexWrapperPattern::Fancy(join_patterns!(
        r"'(?i:[sdmt]|ll|ve|re)",
        r"[^\r\n\p{L}\p{N}]?+\p{L}++",
        r"\p{N}{1,3}+",
        r" ?[^\s\p{L}\p{N}]++[\r\n]*+",
        r"\s++$",
        r"\s*[\r\n]",
        r"\s+(?!\S)",
        r"\s",
    ));

/// The GPT-5 o220k word pattern.
pub const OA_GPT5_O220K_WORD_PATTERN: ConstRegexWrapperPattern = ConstRegexWrapperPattern::Fancy(
    join_patterns!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+"
    ),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        assert!(OA_GPT2_R50K_WORD_PATTERN.compile().is_ok());
        assert!(OA_GPT2_R50K_WORD_PATTERN_SLOW.compile().is_ok());

        assert!(OA_GPT3_CL100K_WORD_PATTERN.compile().is_ok());

        assert!(OA_GPT3_CL100K_WORD_PATTERN.compile().is_ok());
        assert!(OA_GPT5_O220K_WORD_PATTERN.compile().is_ok());
    }
}
