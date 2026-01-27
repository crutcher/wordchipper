//! # Unified Token Vocabulary

use crate::alloc::sync::Arc;
use crate::alloc::vec::Vec;
use crate::segmentation::segmentation_config::SegmentationConfig;
use crate::types::{CommonHashMap, CommonHashSet, Pair, SpanTokenMap, TokenType};
use crate::vocab::ByteMapVocab;
use crate::vocab::pair_vocab::PairMapVocab;
use crate::vocab::span_vocab::SpanMapVocab;
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
        word_vocab: SpanMapVocab<T>,
        pair_vocab: PairMapVocab<T>,
    ) -> Self {
        assert_eq!(word_vocab.byte_vocab(), pair_vocab.byte_vocab());

        let tokens = word_vocab.unordered_tokens().collect::<CommonHashSet<_>>();
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
            span_vocab: word_vocab,
            pair_vocab,
        }
    }

    /// Get the byte table for the word vocabulary.
    ///
    /// ## Returns
    /// A reference to the internal `ByteMapVocab` arc.
    pub fn byte_vocab(&self) -> &Arc<ByteMapVocab<T>> {
        self.span_vocab.byte_vocab()
    }

    /// Extend the vocabulary with the given special words.
    ///
    /// ## Arguments
    /// * `special_words` - An iterator of word strings and tokens.
    ///
    /// ## Returns
    /// The updated `UnifiedTokenVocab` instance.
    pub fn with_special_words<W, S>(
        self,
        special_words: W,
    ) -> Self
    where
        W: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
    {
        Self {
            segmentation: self.segmentation.with_special_words(special_words),
            ..self
        }
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
    ) -> Option<&T> {
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
