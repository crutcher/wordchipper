//! # Word Map ``{ Vec<u8> -> T }`` Token Vocabulary

use crate::alloc::vec::Vec;
use crate::types::{CommonHashMap, TokenType};
use crate::vocab::vocab_types::{ByteTokenMap, SpanTokenMap};
use crate::vocab::{ByteMapVocab, PairMapVocab, PairTokenMap, TokenVocab};

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanMapVocab<T: TokenType> {
    /// The byte/token mapping table.
    pub byte_vocab: ByteMapVocab<T>,

    /// The regex pattern used for text spl
    /// Map of ``{ Vec<u8> -> T }``.
    pub span_map: SpanTokenMap<T>,
}

impl<T: TokenType> Default for SpanMapVocab<T> {
    fn default() -> Self {
        SpanMapVocab::from_byte_vocab(ByteMapVocab::default())
    }
}

impl<T: TokenType> From<SpanTokenMap<T>> for SpanMapVocab<T> {
    fn from(span_map: SpanTokenMap<T>) -> Self {
        Self::from_span_map(span_map)
    }
}

/// Read the ``{ u8 -> T }`` mapping from a ``{ Vec<u8> -> T }`` mapping.
pub fn byte_map_from_span_map<T: TokenType>(span_map: &SpanTokenMap<T>) -> ByteTokenMap<T> {
    span_map
        .iter()
        .filter_map(|(span, &token)| {
            if span.len() == 1 {
                Some((span[0], token))
            } else {
                None
            }
        })
        .collect()
}

/// Validate that a [`ByteMapVocab`] and [`SpanMapVocab`] are compatible.
pub fn try_validate_span_map<T>(
    byte_vocab: &ByteMapVocab<T>,
    span_map: &SpanTokenMap<T>,
) -> anyhow::Result<()>
where
    T: TokenType,
{
    for (span, token) in byte_vocab.span_pairs() {
        let b = span[0];

        if let Some(&map_token) = span_map.get(&span)
            && token != map_token
        {
            anyhow::bail!(
                "ByteTable disagrees with span_map for {b:0x?}: {:?} != {:?}",
                token,
                map_token
            );
        }
    }

    Ok(())
}

impl<T: TokenType> SpanMapVocab<T> {
    /// Build vocabulary from just a [`ByteMapVocab`].
    ///
    /// Will have 255 span entries, each 1-byte long.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    ///
    /// ## Returns
    /// A new `SpanMapVocab` instance.
    ///
    /// ## Panics
    /// Panics if initialization fails.
    pub fn from_byte_vocab(byte_vocab: ByteMapVocab<T>) -> Self {
        let span_map: SpanTokenMap<T> = byte_vocab.span_pairs().collect();

        Self::init(byte_vocab, span_map).unwrap()
    }

    /// Build a [`Self`] from a [`SpanTokenMap`].
    ///
    /// The [`ByteMapVocab`] will be inferred from the [`SpanTokenMap`],
    /// and the default ordinal byte to token mappings.
    ///
    /// ## Arguments
    /// * `span_map` - The span to token mapping.
    ///
    /// ## Returns
    /// A new `SpanMapVocab` instance.
    ///
    /// ## Panics
    /// If the [`ByteMapVocab`] mapping is not 1:1, or if initialization fails.
    pub fn from_span_map(span_map: SpanTokenMap<T>) -> Self {
        let mut byte_map: ByteTokenMap<T> = byte_map_from_span_map(&span_map);
        for ord in 0..256 {
            let b = ord as u8;
            let token = T::from_usize(ord).unwrap();
            byte_map.entry(b).or_insert(token);
        }

        let mut ord_table: Vec<(u8, T)> = byte_map.into_iter().collect();
        ord_table.sort_by_key(|&(k, _)| k);
        let byte_to_token: Vec<T> = ord_table.into_iter().map(|(_, v)| v).collect();

        let byte_vocab: ByteMapVocab<T> = ByteMapVocab::from_byte_to_token(&byte_to_token);

        Self::init(byte_vocab, span_map).unwrap()
    }

    /// Initialize a [`SpanMapVocab`].
    ///
    /// The span map will be the union of the span map,
    /// and all overrides from the `byte_vocab`.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `span_map` - The initial span to token mapping.
    ///
    /// ## Returns
    /// A `Result` containing the new `SpanMapVocab` instance or an error.
    pub fn init(
        byte_vocab: ByteMapVocab<T>,
        mut span_map: SpanTokenMap<T>,
    ) -> anyhow::Result<Self> {
        try_validate_span_map(&byte_vocab, &span_map)?;

        span_map.extend(byte_vocab.span_pairs());

        span_map.shrink_to_fit();

        Ok(Self {
            byte_vocab,
            span_map,
        })
    }

    /// The number of words in the vocabulary.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.span_map.len()
    }

    /// Iterate over the words in the vocabulary.
    ///
    /// ## Returns
    /// An iterator over references to spans and their corresponding tokens.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a Vec<u8>, &'a T)> + 'a {
        self.span_map.iter()
    }

    /// Return the associated token for the word, if any.
    ///
    /// ## Arguments
    /// * `chunk` - The byte slice to look up.
    ///
    /// ## Returns
    /// An `Option` containing the token if the span exists in the vocabulary.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        if chunk.len() == 1 {
            Some(self.byte_vocab.get_token(chunk[0]))
        } else {
            self.span_map.get(chunk).copied()
        }
    }

    /// Build a binary pair map from the word vocabulary.
    ///
    /// ## Returns
    /// A new `PairMapVocab` instance.
    ///
    /// ## Panics
    /// Panics if the generated pair map is invalid.
    pub fn to_pair_vocab(&self) -> PairMapVocab<T> {
        let byte_vocab = self.byte_vocab.clone();

        let mut pairs = PairTokenMap::default();

        let token_to_span: CommonHashMap<T, &[u8]> = self
            .span_map
            .iter()
            .map(|(chunk, &token)| (token, chunk.as_ref()))
            .collect();

        for token in self.unordered_tokens() {
            let span = token_to_span[&token];
            if span.len() <= 1 {
                continue;
            }
            for p in 1..span.len() {
                let pre = &span[..p];
                let post = &span[p..];

                if let Some(a) = self.lookup_token(pre)
                    && let Some(b) = self.lookup_token(post)
                {
                    pairs.insert((a, b), token);
                }
            }
        }

        PairMapVocab::<T>::init(byte_vocab, pairs).unwrap()
    }
}

impl<T: TokenType> TokenVocab<T> for SpanMapVocab<T> {
    fn unordered_tokens(&self) -> impl Iterator<Item = T> {
        self.span_map.values().copied()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_map
            .iter()
            .map(|(chunk, &token)| (chunk.clone(), token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::{ByteMapVocab, TokenVocab};

    #[test]
    fn test_tokens_iter() {
        type T = u32;

        let byte_vocab: ByteMapVocab<T> = Default::default();

        let vocab = SpanMapVocab::<T>::default();

        assert_eq!(vocab.max_token(), byte_vocab.max_token());
        assert_eq!(&vocab.sorted_tokens(), &byte_vocab.sorted_tokens());

        let mut span_map = vocab.span_map.clone();

        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        let vocab = SpanMapVocab::from(span_map);

        assert_eq!(vocab.max_token(), 302);
        assert_eq!(vocab.len(), 256 + 3);

        assert_eq!(
            &vocab.sorted_tokens(),
            &byte_vocab
                .sorted_tokens()
                .into_iter()
                .chain([300_u32, 301, 302].into_iter())
                .collect::<Vec<T>>()
        );
    }

    #[test]
    fn test_lookup_token() {
        type T = u32;

        let mut span_map: SpanTokenMap<T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("a".as_bytes().to_vec(), 301);

        let vocab = SpanMapVocab::<T>::from_span_map(span_map);

        assert_eq!(vocab.lookup_token(b"apple"), Some(300));
        assert_eq!(vocab.lookup_token(b"a"), Some(301));
        assert_eq!(vocab.lookup_token(b"b"), Some('b' as u32));
    }

    #[test]
    fn test_build_pair_vocab() {
        type T = u32;

        let mut span_map: SpanTokenMap<T> = Default::default();
        span_map.insert("at".as_bytes().to_vec(), 300);
        span_map.insert("ate".as_bytes().to_vec(), 301);
        span_map.insert("cat".as_bytes().to_vec(), 302);

        let vocab = SpanMapVocab::from(span_map);

        let pair_vocab = vocab.to_pair_vocab();
        assert_eq!(
            pair_vocab.pairs(),
            &[
                (('a' as u32, 't' as u32), 300),
                ((300, 'e' as u32), 301),
                (('c' as u32, 300), 302)
            ]
            .iter()
            .map(|&(a, b)| (a, b))
            .collect::<PairTokenMap<T>>()
        );
    }
}
