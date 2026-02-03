//! # Unified Token Vocabulary

use crate::alloc::vec::Vec;
use crate::segmentation::segmentation_config::SegmentationConfig;
use crate::types::{CommonHashMap, CommonHashSet, Pair, SpanTokenMap, TokenType};
use crate::vocab::ByteMapVocab;
use crate::vocab::pair_vocab::PairMapVocab;
use crate::vocab::span_vocab::SpanMapVocab;
use crate::vocab::special_vocab::SpecialVocab;
use crate::vocab::token_vocab::TokenVocab;

/// Unified token vocabulary.
#[derive(Clone)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Text Segmentation Configuration
    pub segmentation: SegmentationConfig<T>,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    pub span_vocab: SpanMapVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pub pair_vocab: PairMapVocab<T>,
}

impl<T: TokenType> UnifiedTokenVocab<T> {
    /// Build a new [`UnifiedTokenVocab`] from a [`SpanMapVocab`].
    ///
    /// ## Arguments
    /// * `segmentation` - The segmentation configuration.
    /// * `span_vocab` - The span map vocabulary.
    ///
    /// ## Returns
    /// A new `UnifiedTokenVocab` instance.
    pub fn from_span_vocab(
        segmentation: SegmentationConfig<T>,
        span_vocab: SpanMapVocab<T>,
    ) -> Self {
        let pair_vocab = span_vocab.to_pair_vocab();
        Self::init(segmentation, span_vocab, pair_vocab)
    }

    /// Build a new [`UnifiedTokenVocab`] from a [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `segmentation` - The segmentation configuration.
    /// * `pair_vocab` - The pair map vocabulary.
    ///
    /// ## Returns
    /// A new `UnifiedTokenVocab` instance.
    pub fn from_pair_vocab(
        segmentation: SegmentationConfig<T>,
        pair_vocab: PairMapVocab<T>,
    ) -> Self {
        let word_vocab = pair_vocab.span_pairs().collect::<SpanTokenMap<T>>().into();
        Self::from_span_vocab(segmentation, word_vocab)
    }

    /// Initialize a [`UnifiedTokenVocab`].
    ///
    /// ## Arguments
    /// * `segmentation` - The segmentation configuration.
    /// * `word_vocab` - The span map vocabulary.
    /// * `pair_vocab` - The pair map vocabulary.
    ///
    /// ## Returns
    /// A new `UnifiedTokenVocab` instance.
    ///
    /// ## Panics
    /// Panics if the vocabularies are inconsistent.
    pub fn init(
        segmentation: SegmentationConfig<T>,
        span_vocab: SpanMapVocab<T>,
        pair_vocab: PairMapVocab<T>,
    ) -> Self {
        assert_eq!(&span_vocab.byte_vocab, &pair_vocab.byte_vocab);

        let tokens = span_vocab.unordered_tokens().collect::<CommonHashSet<_>>();
        for ((a, b), c) in pair_vocab.pairs() {
            for t in [a, b, c].iter() {
                assert!(
                    tokens.contains(*t),
                    "pair token {t:?} not found in word vocab"
                );
            }
        }
        for t in segmentation.specials.unordered_tokens() {
            assert!(
                !tokens.contains(&t),
                "special token {t:?} found in word vocab"
            );
        }

        Self {
            segmentation,
            span_vocab,
            pair_vocab,
        }
    }

    /// Get the byte table for the word vocabulary.
    pub fn byte_vocab(&self) -> &ByteMapVocab<T> {
        &self.span_vocab.byte_vocab
    }

    /// Get a reference to the [`SpecialVocab`]
    pub fn special_vocab(&self) -> &SpecialVocab<T> {
        self.segmentation.special_vocab()
    }

    /// Get a mutable view of the [`SpecialVocab`]
    pub fn special_vocab_mut(&mut self) -> &mut SpecialVocab<T> {
        self.segmentation.special_vocab_mut()
    }

    /// Compiled expansion dictionary.
    ///
    /// ## Returns
    /// A hash map from tokens to their corresponding byte vectors.
    pub fn unified_dictionary(&self) -> CommonHashMap<T, Vec<u8>> {
        let mut tmp = CommonHashMap::default();

        self.span_vocab.iter().for_each(|(chunk, &token)| {
            tmp.insert(chunk.clone(), token);
        });

        for (span, token) in self.pair_vocab.span_pairs() {
            if tmp.contains_key(&span) {
                continue;
            }
            tmp.insert(span, token);
        }

        for (span, t) in self.segmentation.special_vocab().span_pairs() {
            tmp.insert(span, t);
        }

        tmp.into_iter()
            .map(|(chunk, token)| (token, chunk))
            .collect()
    }

    /// Looks up a token in the vocabulary using the provided byte slice.
    ///
    /// ## Arguments
    /// * `span` - A byte slice (`&[u8]`) representing the token to be looked up in the vocabulary.
    ///
    /// ## Returns
    /// * `Option<T>` - Returns `Some(T)` if the token is found in the vocabulary,
    ///   where `T` is the type of the value associated with the token. Returns
    ///   `None` if the token is not found.
    pub fn lookup_token(
        &self,
        span: &[u8],
    ) -> Option<T> {
        self.span_vocab.lookup_token(span)
    }

    /// Looks up a given pair in the pair vocabulary and retrieves its associated data, if present.
    ///
    /// ## Arguments
    /// * `pair` - A reference to the `Pair<T>` to be looked up in the pair vocabulary.
    ///
    /// ## Returns
    /// * `Option<&T>` - Returns `Some(&T)` if the pair is found in the pair vocabulary, otherwise returns `None`.
    pub fn lookup_pair(
        &self,
        pair: &Pair<T>,
    ) -> Option<T> {
        self.pair_vocab.lookup_pair(pair)
    }
}

impl<T: TokenType> TokenVocab<T> for UnifiedTokenVocab<T> {
    fn unordered_tokens(&self) -> impl Iterator<Item = T> {
        self.span_vocab
            .unordered_tokens()
            .chain(self.segmentation.specials.unordered_tokens())
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_vocab.span_pairs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        type T = u32;
        let mut span_vocab: SpanMapVocab<T> = Default::default();
        span_vocab.span_map.insert("at".as_bytes().to_vec(), 300);
        span_vocab.span_map.insert("ate".as_bytes().to_vec(), 301);

        let seg_config = SegmentationConfig::from_pattern(r"\w\+");

        let vocab = UnifiedTokenVocab::from_span_vocab(seg_config, span_vocab);
        let byte_vocab = vocab.byte_vocab();

        assert_eq!(vocab.lookup_token("at".as_bytes()), Some(300));
        assert_eq!(vocab.lookup_token("ate".as_bytes()), Some(301));
        assert_eq!(
            vocab.lookup_token("a".as_bytes()),
            Some(byte_vocab.get_token(b'a'))
        );

        assert_eq!(
            vocab.lookup_pair(&(byte_vocab.get_token(b'a'), byte_vocab.get_token(b't'))),
            Some(300)
        );
    }
}
