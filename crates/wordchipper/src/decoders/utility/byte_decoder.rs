//! # Byte Decoder
//!
//! Mainly used for utility.

use crate::decoders::{TokenDecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::byte_vocab::ByteMapVocab;

/// A decoders that only decodes byte tokens.
#[derive(Clone, Default)]
pub struct ByteDecoder<T: TokenType> {
    byte_vocab: ByteMapVocab<T>,
}

impl<T: TokenType> ByteDecoder<T> {
    /// Create a new byte decoder.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    ///
    /// ## Returns
    /// A new `ByteDecoder` instance.
    pub fn new(byte_vocab: ByteMapVocab<T>) -> Self {
        Self { byte_vocab }
    }

    /// Get the byte table.
    ///
    /// ## Returns
    /// A reference to the internal `ByteMapVocab`.
    pub fn byte_vocab(&self) -> &ByteMapVocab<T> {
        &self.byte_vocab
    }
}

impl<T: TokenType> TokenDecoder<T> for ByteDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, ctx)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = self.byte_vocab.get_byte(t) {
                ctx.buf.push(b);
            } else {
                ctx.stack.push(t);
                break;
            }
        }
        ctx.stack.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::vec;

    #[test]
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::new(ByteMapVocab::default());

        let mut tokens = vec![];
        tokens.extend(
            "hello world"
                .as_bytes()
                .iter()
                .map(|&b| decoder.byte_vocab().get_token(b)),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let mut ctx: TokenDecodeContext<T> = tokens.into();
        assert!(!decoder.incremental_decode(&mut ctx));

        assert_eq!(ctx.buf, "hello world".as_bytes().to_vec());
        assert_eq!(ctx.stack, [3000, 256]);
    }
}
