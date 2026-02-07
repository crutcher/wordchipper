//! # Encoder Test Utilities

use crate::alloc::string::String;
use crate::alloc::sync::Arc;
use crate::alloc::vec;
use crate::alloc::vec::Vec;
use crate::compat::slices::inner_slice_view;
use crate::compat::traits::static_is_send_sync_check;
use crate::decoders::{TokenDecoder, TokenDictDecoder};
use crate::encoders::TokenEncoder;
use crate::pretrained::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
use crate::spanning::TextSpanningConfig;
use crate::types::TokenType;
use crate::vocab::utility::testing::{build_test_shift_byte_vocab, build_test_vocab};
use crate::vocab::{TokenVocab, UnifiedTokenVocab};

/// Build common test vocabulary for [`TokenEncoder`] tests.
pub fn common_encoder_test_vocab<T: TokenType>() -> UnifiedTokenVocab<T> {
    let mut vocab: UnifiedTokenVocab<T> = build_test_vocab(
        build_test_shift_byte_vocab(10),
        TextSpanningConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
    );
    let hi_token = vocab.max_token().unwrap() + T::one();
    vocab.special_vocab_mut().add_str_word("<|HI|>", hi_token);

    vocab
}

/// Common [`TokenEncoder`] tests.
pub fn common_encoder_tests<T: TokenType, E: TokenEncoder<T>>(
    vocab: UnifiedTokenVocab<T>,
    encoder: &E,
) {
    static_is_send_sync_check(encoder);

    // Verify:
    // - clone
    // - Arc/Box compat.
    let encoder: Arc<E> = encoder.clone().into();

    let samples = vec![
        "hello world",
        "hello san francisco",
        "it's not the heat, it's the salt",
    ];

    let decoder = TokenDictDecoder::from_unified_vocab(vocab.clone());
    static_is_send_sync_check(&decoder);

    let token_batch = encoder.try_encode_batch(&samples).unwrap();
    let decoded_strings = decoder
        .try_decode_batch_to_strings(&inner_slice_view(&token_batch))
        .unwrap()
        .unwrap();

    assert_eq!(decoded_strings, samples);

    // Build and test a list of all special tokens.

    let specials: Vec<(&[u8], T)> = vocab
        .special_vocab()
        .span_map
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
