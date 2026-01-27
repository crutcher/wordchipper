//! # `crate::vocab::TokenDecoder`] Context

use crate::alloc::vec::Vec;
use crate::types::TokenType;
use crate::vocab::public::size_hints::EXPECTED_BYTES_PER_TOKEN;

/// Representation of a token decoding context.
#[derive(Clone)]
pub struct TokenDecodeContext<T: TokenType> {
    /// Append buffer for decoded bytes.
    pub buf: Vec<u8>,

    /// FILO stack of tokens to be decoded.
    pub stack: Vec<T>,
}

impl<T: TokenType> From<Vec<T>> for TokenDecodeContext<T> {
    fn from(tokens: Vec<T>) -> Self {
        Self::for_tokens_with_hint(tokens, EXPECTED_BYTES_PER_TOKEN)
    }
}

impl<T: TokenType> TokenDecodeContext<T> {
    /// Creates a new decoding context.
    ///
    /// ## Arguments
    /// * `tokens` - the tokens to decode.
    /// * `bytes_per_token_hint` - a hint for the average number of bytes per token,
    ///   used when allocating output buffer space.
    ///
    /// ## Returns
    /// A new `TokenDecodeContext` instance.
    pub fn for_tokens_with_hint(
        tokens: Vec<T>,
        bytes_per_token_hint: f64,
    ) -> Self {
        let capacity = tokens.len() as f64 * bytes_per_token_hint * 1.25;
        let buf = Vec::with_capacity(capacity as usize);
        let mut stack = tokens;
        stack.reverse();
        Self { buf, stack }
    }

    /// The context is complete when the token stack is empty.
    ///
    /// ## Returns
    /// `true` if the token stack is empty, `false` otherwise.
    pub fn is_complete(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns the decoded buffer, or an error if the stack is not empty.
    ///
    /// ## Returns
    /// A `Result` containing the decoded byte vector or an error if decoding is incomplete.
    pub fn try_result(self) -> anyhow::Result<Vec<u8>> {
        if self.is_complete() {
            Ok(self.buf)
        } else {
            Err(anyhow::anyhow!(
                "Incomplete context: [{:?}, ...]",
                self.stack[self.stack.len() - 1]
            ))
        }
    }

    /// Returns the decoded buffer, panics if the stack is not empty.
    ///
    /// ## Returns
    /// The decoded byte vector.
    ///
    /// ## Panics
    /// Panics if the token stack is not empty.
    pub fn unwrap(self) -> Vec<u8> {
        self.try_result().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use crate::alloc::vec;

    #[test]
    fn test_complete() {
        let mut ctx = TokenDecodeContext::<u8>::for_tokens_with_hint(vec![0, 1, 2], 1.0);
        assert!(!ctx.is_complete());

        ctx.stack.pop();
        ctx.buf.push('a' as u8);

        assert!(!ctx.is_complete());

        assert_eq!(
            ctx.clone()
                .try_result()
                .expect_err("expected failure")
                .to_string(),
            anyhow::anyhow!("Incomplete context: [1, ...]").to_string(),
        );

        ctx.stack.pop();
        ctx.buf.push('b' as u8);
        ctx.stack.pop();
        ctx.buf.push('c' as u8);

        assert_eq!(ctx.unwrap(), vec!['a' as u8, 'b' as u8, 'c' as u8]);
    }
}
