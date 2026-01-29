//! # Encoder Test Utilities

use crate::alloc::string::String;
use crate::alloc::sync::Arc;
use crate::alloc::vec;
use crate::alloc::vec::Vec;
use crate::decoders::{DictionaryDecoder, TokenDecoder};
use crate::encoders::TokenEncoder;
use crate::segmentation::SegmentationConfig;
use crate::types::{TokenType, check_is_send, check_is_sync};
use crate::vocab::byte_vocab::build_test_shift_byte_vocab;
use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
use crate::vocab::utility::testing::build_test_vocab;
use crate::vocab::{TokenVocab, UnifiedTokenVocab};

/// Build common test vocabulary for [`TokenEncoder`] tests.
pub fn common_encoder_test_vocab<T: TokenType>() -> Arc<UnifiedTokenVocab<T>> {
    let mut vocab: UnifiedTokenVocab<T> = build_test_vocab(
        build_test_shift_byte_vocab(10),
        SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
    );
    let hi_token = vocab.max_token() + T::one();
    vocab.special_vocab_mut().add_str_word("<|HI|>", hi_token);

    vocab.into()
}

/// Common [`TokenEncoder`] tests.
pub fn common_encoder_tests<T: TokenType, E: TokenEncoder<T>>(
    vocab: Arc<UnifiedTokenVocab<T>>,
    encoder: &E,
) {
    check_is_send(encoder);
    check_is_sync(encoder);

    let samples = vec![
        "hello world",
        "hello san francisco",
        "it's not the heat, it's the salt",
    ];

    let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
    check_is_send(&decoder);
    check_is_sync(&decoder);

    let token_batch = encoder.try_encode_batch(&samples).unwrap();
    let decoded_strings = decoder.try_decode_batch_to_strings(&token_batch).unwrap();

    assert_eq!(decoded_strings, samples);

    // Build and test a list of all special tokens.

    let specials: Vec<(&[u8], T)> = vocab
        .special_vocab()
        .span_map()
        .iter()
        .map(|(span, token)| (span.as_slice(), *token))
        .collect::<Vec<_>>();

    let special_bytes = specials
        .iter()
        .flat_map(|(span, _)| span.to_vec())
        .collect::<Vec<_>>();

    let special_string = String::from_utf8(special_bytes).unwrap();

    let special_tokens = specials.iter().map(|(_, token)| *token).collect::<Vec<_>>();

    assert_eq!(
        encoder.try_encode(special_string.as_str()).unwrap(),
        special_tokens
    );
}
