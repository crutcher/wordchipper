//! # Token Decoder Trait

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::decoders::TokenDecodeContext;
use crate::types::TokenType;
use crate::vocab::utility::strings::string_from_lossy_utf8;

/// Trait for token decoders.
pub trait TokenDecoder<T: TokenType>: Send + Sync {
    /// Incrementally decodes the context.
    ///
    /// Progresses until `ctx.stack` is empty,
    /// or the top token cannot be decoded by this decoders.
    ///
    /// ## Arguments
    /// * `ctx` - The decoding context to process.
    ///
    /// ## Returns
    /// `ctx.stack.is_empty()`
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool;

    /// Decodes tokens into bytes.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `TokenDecodeContext` containing the decoded bytes and any remaining tokens.
    fn try_decode_to_context<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> anyhow::Result<TokenDecodeContext<T>> {
        let mut context = tokens.as_ref().to_vec().into();
        self.incremental_decode(&mut context);
        Ok(context)
    }

    /// Decodes a batch of tokens to [`TokenDecodeContext`].
    ///
    /// ## Arguments
    /// * `batch` - The batch of tokens to decode.
    ///
    /// ## Returns
    /// A `Result` containing the partial decode contexts.
    fn try_decode_batch_to_context<S: AsRef<[T]>>(
        &self,
        batch: &[S],
    ) -> anyhow::Result<Vec<TokenDecodeContext<T>>> {
        batch
            .iter()
            .map(|tokens| self.try_decode_to_context(tokens))
            .collect()
    }

    /// Decode tokens into bytes, returning an error if the decoding fails.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `Result` containing the decoded byte vector or an error if decoding is incomplete.
    fn try_decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> anyhow::Result<Vec<u8>> {
        self.try_decode_to_context(tokens)?.try_result()
    }

    /// Decodes a batch of tokens into a vector of byte vectors, returning an error if the decoding fails.
    ///
    /// ## Arguments
    /// * `batch` - A slice of token vectors to decode.
    ///
    /// ## Returns
    /// A `Result` containing a vector of decoded byte vectors or an error if any decoding fails.
    fn try_decode_batch_to_bytes<V: AsRef<[T]>>(
        &self,
        batch: &[V],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        self.try_decode_batch_to_context(batch)?
            .into_iter()
            .map(|ctx| ctx.try_result())
            .collect()
    }

    /// Decodes tokens into a string, returning an error if the decoding fails.
    ///
    /// UTF-8 lossy decoding is used to handle invalid UTF-8 sequences.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `Result` containing the decoded string or an error if decoding is incomplete.
    fn try_decode_to_string<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> anyhow::Result<String> {
        Ok(string_from_lossy_utf8(self.try_decode_to_bytes(tokens)?))
    }

    /// Decodes a batch of tokens into a vector of strings, returning an error if the decoding fails.
    ///
    /// UTF-8 lossy decoding is used to handle invalid UTF-8 sequences.
    ///
    /// ## Arguments
    /// * `batch` - A slice of token vectors to decode.
    ///
    /// ## Returns
    /// A `Result` containing a vector of decoded strings or an error if any decoding fails.
    fn try_decode_batch_to_strings<V: AsRef<[T]>>(
        &self,
        batch: &[V],
    ) -> anyhow::Result<Vec<String>> {
        self.try_decode_batch_to_bytes(batch).map(|bytes_batch| {
            bytes_batch
                .into_iter()
                .map(string_from_lossy_utf8)
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use crate::alloc::vec;
    use crate::decoders::utility::byte_decoder::ByteDecoder;
    use num_traits::FromPrimitive;

    #[test]
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

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

    #[test]
    fn test_decode_to_strings() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

        let str_samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let token_batch: Vec<Vec<T>> = str_samples
            .iter()
            .map(|s| {
                s.as_bytes()
                    .iter()
                    .map(|b| T::from_u8(*b).unwrap())
                    .collect()
            })
            .collect();

        // Test the batch interfaces.
        let string_batch = decoder.try_decode_batch_to_strings(&token_batch).unwrap();
        assert_eq!(string_batch, str_samples);

        // Test the single-sample interfaces.
        for (sample, tokens) in str_samples.iter().zip(token_batch.iter()) {
            assert_eq!(
                decoder.try_decode_to_string(tokens).unwrap(),
                sample.to_string()
            );
        }
    }
}
