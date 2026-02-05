//! # Byte Decoder
//!
//! Mainly used for utility.

use crate::alloc::vec::Vec;
use crate::decoders::decode_results::DecodeResult;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::TokenType;
use crate::vocab::byte_vocab::ByteMapVocab;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;

/// A [`ByteMapVocab`] based [`TokenDecoder`].
///
/// Can only decode the byte tokens,
/// useful primarily for testing.
#[derive(Clone, Default)]
pub struct ByteDecoder<T: TokenType> {
    /// The byte vocabulary mapping.
    pub byte_vocab: ByteMapVocab<T>,
}

impl<T: TokenType> From<ByteMapVocab<T>> for ByteDecoder<T> {
    fn from(byte_vocab: ByteMapVocab<T>) -> Self {
        Self::init(byte_vocab)
    }
}

impl<T: TokenType> ByteDecoder<T> {
    /// Create a new byte decoder.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    ///
    /// ## Returns
    /// A new `ByteDecoder` instance.
    pub fn init(byte_vocab: ByteMapVocab<T>) -> Self {
        Self { byte_vocab }
    }
}

impl<T: TokenType> TokenDecoder<T> for ByteDecoder<T> {
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<Vec<u8>>> {
        let capacity = (tokens.len() as f64 * EXPECTED_BYTES_PER_TOKEN) as usize;
        let mut value = Vec::with_capacity(capacity);
        let mut consumed = 0;
        for &t in tokens {
            if let Some(b) = self.byte_vocab.get_byte(t) {
                value.push(b);
                consumed += 1;
            } else {
                break;
            }
        }
        Ok(DecodeResult::new(value, Some(tokens.len() - consumed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::vec;

    #[test]
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteMapVocab::default().into();

        let mut tokens = vec![];
        tokens.extend(
            "hello world"
                .as_bytes()
                .iter()
                .map(|&b| decoder.byte_vocab.get_token(b)),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let result = decoder.try_decode_to_bytes(&tokens).unwrap();
        assert_eq!(result.value, "hello world".as_bytes().to_vec());
        assert_eq!(result.remaining, Some(2));
    }
}
