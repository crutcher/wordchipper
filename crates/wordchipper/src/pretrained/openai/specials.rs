//! # Special Tokens

use crate::alloc::string::String;
use crate::alloc::string::ToString;
use crate::alloc::vec::Vec;
use crate::declare_carrot_special;
use crate::vocab::utility::specials_tools::format_reserved_carrot;

declare_carrot_special!(
    (STARTOFTEXT, "startoftext"),
    (ENDOFTEXT, "endoftext"),
    (ENDOFPROMPT, "endofprompt"),
    (FIM_PREFIX, "fim_prefix"),
    (FIM_MIDDLE, "fim_middle"),
    (FIM_SUFFIX, "fim_suffix"),
    (RETURN, "return"),
    (CONSTRAIN, "constrain"),
    (CHANNEL, "channel"),
    (START, "start"),
    (END, "end"),
    (MESSAGE, "message"),
    (CALL, "call"),
);

/// The GPT-2 "r50k" special tokens.
pub const OA_GPT2_R50K_SPECIALS: &[(&str, usize)] = &[(ENDOFTEXT, 50256)];

/// The GPT-2 "r50k" special tokens.
pub fn oa_gpt2_r50k_specials() -> Vec<(String, usize)> {
    OA_GPT2_R50K_SPECIALS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect()
}

/// The GPT-2 "p50k base" special tokens.
pub const OA_GPT2_P50K_BASE_SPECIALS: &[(&str, usize)] = &[(ENDOFTEXT, 50256)];

/// The GPT-2 "p50k base" special tokens.
pub fn oa_gpt2_p50k_base_specials() -> Vec<(String, usize)> {
    OA_GPT2_P50K_BASE_SPECIALS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect()
}

/// The GPT-2 "p50k edit" special tokens.
pub const OA_GPT2_P50K_EDIT_SPECIALS: &[(&str, usize)] = &[
    (ENDOFTEXT, 50256),
    (FIM_PREFIX, 50281),
    (FIM_MIDDLE, 50282),
    (FIM_SUFFIX, 50283),
];

/// The GPT-2 "p50k edit" special tokens.
pub fn oa_gpt2_p50k_edit_specials() -> Vec<(String, usize)> {
    OA_GPT2_P50K_EDIT_SPECIALS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect()
}

/// The GPT-3 "cl100k" special tokens.
pub const OA_GPT3_CL100K_EDIT_SPECIALS: &[(&str, usize)] = &[
    (ENDOFTEXT, 100257),
    (FIM_PREFIX, 100258),
    (FIM_MIDDLE, 100259),
    (FIM_SUFFIX, 100260),
    (ENDOFPROMPT, 100276),
];

/// The GPT-3 "cl100k" special tokens.
pub fn oa_gpt3_cl100k_edit_specials() -> Vec<(String, usize)> {
    OA_GPT3_CL100K_EDIT_SPECIALS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect()
}

/// The GPT-5 "o200k base" special tokens.
pub const OA_GPT5_O200K_BASE_SPECIALS: &[(&str, usize)] =
    &[(ENDOFTEXT, 199999), (ENDOFPROMPT, 200018)];

/// The GPT-5 "o200k base" special tokens.
pub fn oa_gt5_o200k_base_specials() -> Vec<(String, usize)> {
    OA_GPT5_O200K_BASE_SPECIALS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect()
}

/// The GPT-5 "o200k harmony" special tokens.
pub const OA_GPT5_O200K_HARMONY_NAMED_SPECIALS: &[(&str, usize)] = &[
    (STARTOFTEXT, 199998),
    (ENDOFTEXT, 199999),
    (ENDOFPROMPT, 200018),
    (RETURN, 200002),
    (CONSTRAIN, 200003),
    (CHANNEL, 200005),
    (START, 200006),
    (END, 200007),
    (MESSAGE, 200008),
    (CALL, 200012),
];

/// The GPT-5 "o200k harmony" named special tokens; excluding reserved tokens.
pub fn oa_gpt5_o200k_harmony_named_specials() -> Vec<(String, usize)> {
    OA_GPT5_O200K_HARMONY_NAMED_SPECIALS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect()
}

/// Generate the GPT-5 "o200k harmony" reserved tokens.
pub fn oa_gpt5_o200k_harmony_gen_reserved() -> Vec<(String, usize)> {
    let mut specials: Vec<(String, usize)> = Vec::with_capacity(6 + (201088 - 200013));

    let mut reserve = |val| {
        specials.push((format_reserved_carrot(val), val));
    };

    reserve(200000);
    reserve(200001);
    reserve(200004);
    reserve(200009);
    reserve(200010);
    reserve(200011);

    for val in 200013..201088 {
        reserve(val);
    }

    specials
}

/// The GPT-5 "o200k harmony" special tokens.
///
/// Generated due to the large number of reserved tokens.
pub fn oa_gpt5_o200k_harmony_specials() -> Vec<(String, usize)> {
    let mut specials = oa_gpt5_o200k_harmony_named_specials();

    specials.extend(oa_gpt5_o200k_harmony_gen_reserved());

    specials
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::vec;

    #[test]
    fn test_oa_gpt2_r50k_specials() {
        assert_eq!(
            oa_gpt2_r50k_specials(),
            vec![("<|endoftext|>".to_string(), 50256),]
        );
    }

    #[test]
    fn test_oa_gpt2_p50k_base_specials() {
        assert_eq!(
            oa_gpt2_p50k_base_specials(),
            vec![("<|endoftext|>".to_string(), 50256),]
        );
    }

    #[test]
    fn test_oa_gpt2_p50k_edit_specials() {
        assert_eq!(
            oa_gpt2_p50k_edit_specials(),
            vec![
                ("<|endoftext|>".to_string(), 50256),
                ("<|fim_prefix|>".to_string(), 50281),
                ("<|fim_middle|>".to_string(), 50282),
                ("<|fim_suffix|>".to_string(), 50283),
            ]
        );
    }

    #[test]
    fn test_oa_gpt3_cl100k_edit_specials() {
        assert_eq!(
            oa_gpt3_cl100k_edit_specials(),
            vec![
                ("<|endoftext|>".to_string(), 100257),
                ("<|fim_prefix|>".to_string(), 100258),
                ("<|fim_middle|>".to_string(), 100259),
                ("<|fim_suffix|>".to_string(), 100260),
                ("<|endofprompt|>".to_string(), 100276),
            ]
        );
    }

    #[test]
    fn test_oa_gpt5_o200k_base_specials() {
        assert_eq!(
            oa_gt5_o200k_base_specials(),
            vec![
                ("<|endoftext|>".to_string(), 199999),
                ("<|endofprompt|>".to_string(), 200018)
            ]
        )
    }

    #[test]
    fn test_oa_gpt5_o200k_harmony_specials() {
        let mut expected = vec![
            ("<|reserved_200000|>".to_string(), 200000),
            ("<|reserved_200001|>".to_string(), 200001),
            ("<|reserved_200004|>".to_string(), 200004),
            ("<|reserved_200009|>".to_string(), 200009),
            ("<|reserved_200010|>".to_string(), 200010),
            ("<|reserved_200011|>".to_string(), 200011),
        ];
        (200013..201088).for_each(|i| expected.push((format_reserved_carrot(i), i)));

        let reserved = oa_gpt5_o200k_harmony_gen_reserved();
        assert_eq!(&reserved, &expected);

        let named = oa_gpt5_o200k_harmony_named_specials();
        assert_eq!(
            &named,
            &vec![
                ("<|startoftext|>".to_string(), 199998),
                ("<|endoftext|>".to_string(), 199999),
                ("<|endofprompt|>".to_string(), 200018),
                ("<|return|>".to_string(), 200002),
                ("<|constrain|>".to_string(), 200003),
                ("<|channel|>".to_string(), 200005),
                ("<|start|>".to_string(), 200006),
                ("<|end|>".to_string(), 200007),
                ("<|message|>".to_string(), 200008),
                ("<|call|>".to_string(), 200012),
            ]
        );

        let expected = named
            .iter()
            .chain(reserved.iter())
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(oa_gpt5_o200k_harmony_specials(), expected);
    }
}
