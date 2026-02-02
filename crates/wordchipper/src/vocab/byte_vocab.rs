//! # Byte/Token Mapping Table

use crate::alloc::vec;
use crate::alloc::vec::Vec;
use crate::types::{CommonHashMap, TokenType};
use crate::vocab::TokenVocab;
use core::fmt::Debug;

/// Build a [`ByteMapVocab`] with all tokens shifted by `shift`.
///
/// This is a purposely stupid byte map; useful for testing.
pub fn build_test_shift_byte_vocab<T: TokenType>(shift: usize) -> ByteMapVocab<T> {
    // This is a purposely stupid byte map.
    ByteMapVocab::<T>::from_byte_to_token(
        &ByteMapVocab::<T>::default()
            .byte_to_token()
            .iter()
            .map(|&t| t + T::from_usize(shift).unwrap())
            .collect::<Vec<T>>(),
    )
}

/// ``0..=255`` Rank Byte/Token Bijection Table
///
/// This will always have 255 entries, one for each byte value.
/// The token values are not required to be dense, or in the range ``0..=255``.
/// This is required to be a bijection (255 distinct tokens).
#[derive(Clone, PartialEq)]
pub struct ByteMapVocab<T: TokenType> {
    /// Hash map from token to byte ordinal value.
    pub token_to_byte: CommonHashMap<T, u8>,

    /// Table mapping from byte ordinal (position) to token.
    pub byte_to_token: [T; 256],
}

impl<T: TokenType> Debug for ByteMapVocab<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("ByteTable")
            .field("max_token", &self.max_token())
            .field("tokens", &self.token_to_byte)
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

        let mut token_to_byte: CommonHashMap<T, u8> = byte_to_token
            .iter()
            .enumerate()
            .map(|(t, &token)| (token, t as u8))
            .collect();
        token_to_byte.shrink_to_fit();

        assert_eq!(token_to_byte.len(), 256);

        Self {
            token_to_byte,
            byte_to_token,
        }
    }

    /// Build a `ByteTable` from a token => byte hash map.
    ///
    /// ## Arguments
    /// * `token_to_byte` - A hash map from token to byte value.
    ///
    /// ## Returns
    /// A new `ByteMapVocab` instance.
    ///
    /// ## Panics
    /// If the map is not a 1:1 bijection.
    pub fn from_token_to_byte(token_to_byte: &CommonHashMap<T, u8>) -> Self {
        let token_to_byte = token_to_byte.clone();

        let ord_map: CommonHashMap<u8, T> = token_to_byte.iter().map(|(&t, &b)| (b, t)).collect();
        assert_eq!(ord_map.len(), 256);

        let mut ord_items = ord_map.into_iter().collect::<Vec<_>>();
        ord_items.sort_by_key(|(b, _)| *b);

        let byte_to_token: [T; 256] = ord_items
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            byte_to_token,
            token_to_byte,
        }
    }

    /// Get the length of the table.
    ///
    /// ## Returns
    /// The number of entries in the table (always 256).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.byte_to_token.len()
    }

    /// Get the byte-ord => token mapping table.
    ///
    /// ## Returns
    /// A reference to the fixed-size array mapping bytes to tokens.
    pub fn byte_to_token(&self) -> &[T; 256] {
        &self.byte_to_token
    }

    /// Get the token->byte hash map.
    ///
    /// ## Returns
    /// A reference to the internal hash map mapping tokens to bytes.
    pub fn token_to_byte(&self) -> &CommonHashMap<T, u8> {
        &self.token_to_byte
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
        self.byte_to_token[byte as usize]
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
    pub fn get_byte(
        &self,
        token: T,
    ) -> Option<u8> {
        self.token_to_byte.get(&token).copied()
    }
}

impl<T: TokenType> TokenVocab<T> for ByteMapVocab<T> {
    fn unordered_tokens(&self) -> impl Iterator<Item = T> {
        self.byte_to_token.iter().copied()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.byte_to_token
            .iter()
            .enumerate()
            .map(|(idx, &token)| (vec![idx as u8], token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::format;

    #[test]
    fn test_byte_vocab_default() {
        type T = u32;
        let table: ByteMapVocab<T> = ByteMapVocab::default();

        assert_eq!(table.len(), 256);

        assert_eq!(
            format!("{:?}", table),
            format!(
                "ByteTable {{ max_token: 255, tokens: {:?} }}",
                table.token_to_byte
            )
        );

        for idx in 0..256 {
            let byte = idx as u8;
            let token = idx as u32;

            assert_eq!(table.get_token(byte), token);
            assert_eq!(table.byte_to_token()[idx], token);
            assert_eq!(table.get_byte(token), Some(byte));
            assert_eq!(table.token_to_byte()[&token], byte);
        }

        let rebuild = ByteMapVocab::from_token_to_byte(&table.token_to_byte());
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
