//! # Common Decoder Unit Tests

use crate::alloc::vec;
use crate::alloc::vec::Vec;
use crate::compat::strings::string_from_utf8_lossy;
use crate::decoders::TokenDecoder;
use crate::encoders::{DefaultTokenEncoder, TokenEncoder};
use crate::types::{TokenType, check_is_send, check_is_sync};
use crate::vocab::{TokenVocab, UnifiedTokenVocab};

/// Common Unittest for TokenDecoder implementations.
pub fn common_decoder_unit_test<T: TokenType, D: TokenDecoder<T>>(
    vocab: UnifiedTokenVocab<T>,
    decoder: &D,
) {
    check_is_send(decoder);
    check_is_sync(decoder);

    let samples = vec![
        "hello world",
        "hello san francisco",
        "it's not the heat, it's the salt",
    ];

    let encoder = DefaultTokenEncoder::<T>::init(vocab.clone(), None);

    let token_batch = encoder.try_encode_batch(&samples).unwrap();
    let decoded_strings = decoder.try_decode_batch_to_strings(&token_batch).unwrap();

    assert_eq!(
        &decoder
            .try_decode_batch_to_bytes(&token_batch)
            .unwrap()
            .into_iter()
            .map(string_from_utf8_lossy)
            .collect::<Vec<_>>(),
        &decoded_strings
    );

    assert_eq!(decoded_strings, samples);

    let novel_token = vocab.max_token() + T::one();
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
        .try_decode_batch_to_context(&partial_tokens)
        .unwrap();

    let mut expected_stack = broken_tail.clone();
    expected_stack.reverse();

    for (idx, sample) in samples.iter().enumerate() {
        let ctx = partial_decode.get(idx).unwrap();

        assert!(!ctx.is_complete());
        assert_eq!(ctx.buf, sample.as_bytes().to_vec());
        assert_eq!(&ctx.stack, &expected_stack);
    }
}
