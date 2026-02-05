//! # Token Decoder Trait

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::compat::strings::string_from_utf8_lossy;
use crate::types::TokenType;

/// The result of decoding tokens into bytes.
pub struct DecodeResult<V> {
    /// The remaining token count.
    pub remaining: Option<usize>,

    /// The decoded result.
    pub value: V,
}

impl<V> DecodeResult<V> {
    /// Construct a new result.
    pub fn new(
        value: V,
        remaining: Option<usize>,
    ) -> Self {
        let remaining = remaining.filter(|&r| r > 0);
        Self { value, remaining }
    }

    /// Try to unwrap the result, returning an error if the decoding is incomplete.
    pub fn try_result(self) -> anyhow::Result<V> {
        if let Some(remaining) = self.remaining
            && remaining > 0
        {
            return Err(anyhow::anyhow!(
                "Incomplete decode: {} remaining tokens",
                remaining
            ));
        }
        Ok(self.value)
    }

    /// Unwrap the result, panicking if the decoding is incomplete.
    pub fn unwrap(self) -> V {
        self.try_result().unwrap()
    }

    /// Returns `true` if the decoding is complete.
    pub fn is_complete(&self) -> bool {
        if let Some(remaining) = self.remaining
            && remaining > 0
        {
            return false;
        }
        true
    }

    /// Convert the result using a conversion function.
    pub fn convert<F, U>(
        self,
        f: F,
    ) -> DecodeResult<U>
    where
        F: Fn(V) -> U,
    {
        DecodeResult {
            remaining: self.remaining,
            value: f(self.value),
        }
    }
}

/// The result of decoding a batch of tokens into bytes.
pub struct BatchDecodeResult<V> {
    /// The per-item results.
    pub results: Vec<DecodeResult<V>>,
}

impl<V> From<Vec<DecodeResult<V>>> for BatchDecodeResult<V> {
    fn from(results: Vec<DecodeResult<V>>) -> Self {
        Self { results }
    }
}

impl<V> BatchDecodeResult<V> {
    /// Is the decoding complete for all items?
    pub fn is_complete(&self) -> bool {
        self.results.iter().all(|r| r.is_complete())
    }

    /// Try to unwrap the results, returning an error if any decoding is incomplete.
    pub fn try_results(self) -> anyhow::Result<Vec<V>> {
        self.results.into_iter().map(|r| r.try_result()).collect()
    }

    /// Unwrap the results, panicking if any decoding is incomplete.
    pub fn unwrap(self) -> Vec<V> {
        self.try_results().unwrap()
    }

    /// Convert the results using a conversion function.
    pub fn convert<F, U>(
        self,
        f: &F,
    ) -> BatchDecodeResult<U>
    where
        F: Fn(V) -> U,
    {
        BatchDecodeResult {
            results: self.results.into_iter().map(|r| r.convert(f)).collect(),
        }
    }
}

/// Trait for token decoders.
pub trait TokenDecoder<T: TokenType>: Send + Sync {
    /// Decodes tokens into bytes.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `Result<DecodeResult<Vec<u8>>>`.
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<Vec<u8>>>;

    /// Decodes a batch of tokens.
    ///
    /// ## Arguments
    /// * `batch` - A batch of tokens.
    ///
    /// ## Returns
    /// A `Result<Vec<DecodeResult<Vec<u8>>>>`.
    fn try_decode_batch_to_bytes(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<BatchDecodeResult<Vec<u8>>> {
        batch
            .iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect::<anyhow::Result<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }

    /// Decodes tokens into a string.
    ///
    /// UTF-8 lossy decoding is used to handle invalid UTF-8 sequences.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `Result<Vec<DecodeResult<String>>>`.
    fn try_decode_to_string(
        &self,
        tokens: &[T],
    ) -> anyhow::Result<DecodeResult<String>> {
        self.try_decode_to_bytes(tokens)
            .map(|res| res.convert(string_from_utf8_lossy))
    }

    /// Decodes a batch of tokens.
    ///
    /// UTF-8 lossy decoding is used to handle invalid UTF-8 sequences.
    ///
    /// ## Arguments
    /// * `batch` - A batch of tokens.
    ///
    /// ## Returns
    /// A `Result<Vec<(usize, String)>>` of `[(consumed takens, string)]`.
    fn try_decode_batch_to_strings(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<BatchDecodeResult<String>> {
        batch
            .iter()
            .map(|tokens| self.try_decode_to_string(tokens))
            .collect::<anyhow::Result<Vec<_>>>()
            .map(BatchDecodeResult::from)
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
                .map(|&b| decoder.byte_vocab.get_token(b)),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let result = decoder.try_decode_to_bytes(&tokens).unwrap();
        assert_eq!(result.value, "hello world".as_bytes().to_vec());
        assert_eq!(result.remaining, Some(2));
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
        let string_batch = decoder
            .try_decode_batch_to_strings(
                &token_batch
                    .iter()
                    .map(|v| v.as_ref())
                    .collect::<Vec<&[T]>>(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(string_batch, str_samples);

        // Test the single-sample interfaces.
        for (sample, tokens) in str_samples.iter().zip(token_batch.iter()) {
            assert_eq!(
                decoder.try_decode_to_string(tokens).unwrap().unwrap(),
                sample.to_string()
            );
        }
    }
}
