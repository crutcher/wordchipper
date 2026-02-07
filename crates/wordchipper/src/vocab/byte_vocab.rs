//! # Byte/Token Mapping Table

use crate::alloc::vec;
use crate::alloc::vec::Vec;
use crate::types::TokenType;
use crate::vocab::utility::validators::try_vocab_size;
use crate::vocab::{ByteTokenArray, ByteTokenMap, TokenByteMap, TokenVocab};
use core::fmt::Debug;

/// ``0..=255`` Rank Byte/Token Bijection Table
///
/// This will always have 255 entries, one for each byte value.
/// The token values are not required to be dense, or in the range ``0..=255``.
/// This is required to be a bijection (255 distinct tokens).
#[derive(Clone, PartialEq)]
pub struct ByteMapVocab<T: TokenType> {
    /// Hash map from token to byte ordinal value.
    pub token_bytes: TokenByteMap<T>,

    /// Table mapping from byte ordinal (position) to token.
    pub byte_tokens: [T; 256],
}

impl<T: TokenType> Debug for ByteMapVocab<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("ByteTable")
            .field("max_token", &self.max_token())
            .field("tokens", &self.token_bytes)
            .finish()
    }
}

impl<T: TokenType> Default for ByteMapVocab<T> {
    fn default() -> Self {
        let byte_to_token = (0..256)
            .map(|i| T::from_usize(i).unwrap())
            .collect::<Vec<_>>();
        Self::from_byte_to_token(&byte_to_token)
    }
}

impl<T: TokenType> ByteMapVocab<T> {
    /// Build a `ByteTable` from a byte-ord => token table.
    ///
    /// ## Arguments
    /// * `byte_to_token` - A slice of tokens where the index corresponds to the byte value.
    ///
    /// ## Returns
    /// A new `ByteMapVocab` instance.
    ///
    /// ## Panics
    /// If the map is not a 1:1 bijection.
    pub fn from_byte_to_token(byte_to_token: &[T]) -> Self {
        assert_eq!(byte_to_token.len(), 256);

        let byte_to_token: [T; 256] = byte_to_token.try_into().unwrap();

        let mut token_to_byte: TokenByteMap<T> = byte_to_token
            .iter()
            .enumerate()
            .map(|(t, &token)| (token, t as u8))
            .collect();
        token_to_byte.shrink_to_fit();

        assert_eq!(token_to_byte.len(), 256);

        Self {
            token_bytes: token_to_byte,
            byte_tokens: byte_to_token,
        }
    }

    /// Build a `ByteTable` from a token => byte hash map.
    ///
    /// ## Arguments
    /// * `token_byte_map` - A hash map from token to byte value.
    ///
    /// ## Returns
    /// A new `ByteMapVocab` instance.
    ///
    /// ## Panics
    /// If the map is not a 1:1 bijection.
    pub fn from_token_byte_map(token_byte_map: &TokenByteMap<T>) -> Self {
        let token_bytes = token_byte_map.clone();

        let ord_map: ByteTokenMap<T> = token_bytes.iter().map(|(&t, &b)| (b, t)).collect();
        assert_eq!(ord_map.len(), 256);

        let mut ord_items = ord_map.into_iter().collect::<Vec<_>>();
        ord_items.sort_by_key(|(b, _)| *b);

        let byte_tokens: [T; 256] = ord_items
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            byte_tokens,
            token_bytes,
        }
    }

    /// Convert to a different token type.
    pub fn to_token_type<G: TokenType>(&self) -> anyhow::Result<ByteMapVocab<G>> {
        try_vocab_size::<G>(self.max_token().unwrap().to_usize().unwrap())?;

        Ok(ByteMapVocab::<G>::from_byte_to_token(
            &self
                .byte_tokens
                .into_iter()
                .map(|t| G::from_usize(t.to_usize().unwrap()).unwrap())
                .collect::<Vec<_>>(),
        ))
    }

    /// Get the length of the table.
    ///
    /// ## Returns
    /// The number of entries in the table (always 256).
    pub fn len(&self) -> usize {
        self.byte_tokens.len()
    }

    /// Is this empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the byte-ord => token mapping table.
    ///
    /// ## Returns
    /// A reference to the fixed-size array mapping bytes to tokens.
    pub fn byte_tokens(&self) -> &ByteTokenArray<T> {
        &self.byte_tokens
    }

    /// Get the token->byte hash map.
    ///
    /// ## Returns
    /// A reference to the internal hash map mapping tokens to bytes.
    pub fn token_bytes(&self) -> &TokenByteMap<T> {
        &self.token_bytes
    }

    /// Get the token corresponding to a given byte.
    ///
    /// ## Arguments
    /// * `byte` - The byte value to look up.
    ///
    /// ## Returns
    /// The token corresponding to the byte.
    #[inline(always)]
    pub fn get_token(
        &self,
        byte: u8,
    ) -> T {
        self.byte_tokens[byte as usize]
    }

    /// Append the translated byte tokens to a target buffer.
    ///
    /// ## Arguments
    /// * `bytes` - The slice of bytes to translate and append.
    /// * `tokens` - The target token buffer.
    #[inline(always)]
    pub fn append_tokens(
        &self,
        bytes: &[u8],
        tokens: &mut Vec<T>,
    ) {
        tokens.extend(bytes.iter().map(|&b| self.get_token(b)));
    }

    /// Get the byte corresponding to a given token, if any.
    ///
    /// ## Arguments
    /// * `token` - The token to look up.
    ///
    /// ## Returns
    /// An `Option` containing the byte value if it exists in the mapping.
    #[inline(always)]
    pub fn get_byte(
        &self,
        token: T,
    ) -> Option<u8> {
        self.token_bytes.get(&token).copied()
    }
}

impl<T: TokenType> TokenVocab<T> for ByteMapVocab<T> {
    type Token = T;

    fn tokens(&self) -> Vec<T> {
        let mut tokens = self.byte_tokens.to_vec();
        tokens.sort_unstable();
        tokens
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.byte_tokens
            .iter()
            .enumerate()
            .map(|(idx, &token)| (vec![idx as u8], token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::format;
    use crate::vocab::utility::testing::build_test_shift_byte_vocab;

    #[test]
    fn test_byte_vocab_default() {
        type T = u32;
        let table: ByteMapVocab<T> = ByteMapVocab::default();

        assert_eq!(table.len(), 256);

        assert_eq!(
            format!("{:?}", table),
            format!(
                "ByteTable {{ max_token: Some(255), tokens: {:?} }}",
                table.token_bytes
            )
        );

        for idx in 0..256 {
            let byte = idx as u8;
            let token = idx as u32;

            assert_eq!(table.get_token(byte), token);
            assert_eq!(table.byte_tokens()[idx], token);
            assert_eq!(table.get_byte(token), Some(byte));
            assert_eq!(table.token_bytes()[&token], byte);
        }

        let rebuild = ByteMapVocab::from_token_byte_map(&table.token_bytes());
        assert_eq!(rebuild, table);
    }

    #[test]
    fn test_byte_vocab() {
        type T = u32;

        let vocab = build_test_shift_byte_vocab::<T>(100);

        assert_eq!(vocab.get_token(0_u8), 100);
        assert_eq!(vocab.get_token(255_u8), 355);

        assert_eq!(vocab.get_byte(99_u32), None);
        assert_eq!(vocab.get_byte(100_u32), Some(0));
        assert_eq!(vocab.get_byte(355_u32), Some(255));
        assert_eq!(vocab.get_byte(356_u32), None);
    }
}
