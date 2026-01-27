//! Validators for various configuration options.
use crate::types::TokenType;

/// The size of the u8 space.
pub const U8_SIZE: usize = u8::MAX as usize + 1;

/// Validates and returns the vocabulary size, ensuring it's at least the size of the u8 space.
pub fn try_vocab_size<T: TokenType>(vocab_size: usize) -> anyhow::Result<usize> {
    if T::from_usize(vocab_size).is_none() {
        Err(anyhow::anyhow!(
            "vocab_size ({}) doesn't fit in TokenType: {}",
            vocab_size,
            core::any::type_name::<T>()
        ))
    } else if vocab_size < U8_SIZE {
        Err(anyhow::anyhow!(
            "vocab_size ({vocab_size}) must be >= 256 (the size of the u8 space)"
        ))
    } else {
        Ok(vocab_size)
    }
}

/// Validates and returns the vocab size, panicking if it's too small.
pub fn expect_vocab_size<T: TokenType>(vocab_size: usize) -> usize {
    try_vocab_size::<T>(vocab_size).unwrap()
}
