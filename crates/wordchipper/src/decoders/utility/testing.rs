//! # Common Decoder Unit Tests

use crate::{
    TokenEncoderBuilder,
    alloc::{vec, vec::Vec},
    compat::{strings::string_from_utf8_lossy, traits::static_is_send_sync_check},
    decoders::TokenDecoder,
    types::TokenType,
    vocab::{UnifiedTokenVocab, VocabIndex},
};

/// Common Unittest for TokenDecoder implementations.
pub fn common_decoder_unit_test<T: TokenType, D: TokenDecoder<T>>(
    vocab: UnifiedTokenVocab<T>,
    decoder: &D,
) {
    static_is_send_sync_check(decoder);

    let samples = vec![
        "hello world",
        "hello san francisco",
        "it's not the heat, it's the salt",
    ];

    let encoder = TokenEncoderBuilder::<T>::new(vocab.clone())
        .with_parallel(false)
        .init();

    let token_batch = encoder.try_encode_batch(&samples).unwrap();
    let decoded_strings = decoder
        .try_decode_batch_to_strings(
            &token_batch
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<&[T]>>(),
        )
        .unwrap()
        .unwrap();

    assert_eq!(
        &decoder
            .try_decode_batch_to_bytes(
                &token_batch
                    .iter()
                    .map(|v| v.as_ref())
                    .collect::<Vec<&[T]>>()
            )
            .unwrap()
            .unwrap()
            .into_iter()
            .map(string_from_utf8_lossy)
            .collect::<Vec<_>>(),
        &decoded_strings
    );

    assert_eq!(decoded_strings, samples);

    let novel_token = vocab.max_token().unwrap() + T::one();
    let mut broken_tail = vec![novel_token];
    encoder.try_encode_append("abc", &mut broken_tail).unwrap();

    // Partial Decode
    let partial_tokens = token_batch
        .iter()
        .map(|tokens| {
            let mut extended_tokens = tokens.clone();
            extended_tokens.extend_from_slice(&broken_tail);
            extended_tokens
        })
        .collect::<Vec<_>>();

    let partial_decode = decoder
        .try_decode_batch_to_bytes(
            &partial_tokens
                .iter()
                .map(|v| v.as_ref())
                .collect::<Vec<&[T]>>(),
        )
        .unwrap();

    let mut expected_stack = broken_tail.clone();
    expected_stack.reverse();

    for (idx, sample) in samples.iter().enumerate() {
        let ctx = &partial_decode.results[idx];
        assert!(!ctx.is_complete());
        assert_eq!(ctx.value, sample.as_bytes().to_vec());
        assert_eq!(ctx.remaining, Some(expected_stack.len()));
    }
}
