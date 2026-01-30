//! # Tokenizer Timer Utils.

use wordchipper::decoders::TokenDecoder;
use wordchipper::encoders::TokenEncoder;
use wordchipper::types::TokenType;

/// Struct holding a [`TokenEncoder`] and [`TokenDecoder`].
pub struct FullMontyTokenizer<T, E, D>
where
    T: TokenType,
    E: TokenEncoder<T>,
    D: TokenDecoder<T>,
{
    pub encoder: E,
    pub decoder: D,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, E, D> FullMontyTokenizer<T, E, D>
where
    T: TokenType,
    E: TokenEncoder<T>,
    D: TokenDecoder<T>,
{
    /// Create a new `FullMontyTokenizer`.
    pub fn init(
        encoder: E,
        decoder: D,
    ) -> Self {
        Self {
            encoder,
            decoder,
            _phantom: Default::default(),
        }
    }
}
