//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::alloc::vec::Vec;
use crate::decoders::TokenDecoder;
use crate::decoders::utility::PairExpansionDecoder;
use crate::types::{CommonHashSet, Pair, TokenType};
use crate::vocab::byte_vocab::ByteMapVocab;
use crate::vocab::token_vocab::TokenVocab;
use crate::vocab::vocab_types::PairTokenMap;

/// Validate that a [`ByteMapVocab`] and [`PairTokenMap`] are compatible.
///
/// - for every ``(a, b) -> t`` entry:
///   - the parents ``(a, b)``:
///     - are either in the `byte_vocab`, or are targets in the map, not both.
///   - the target ``t`` is not in the `byte_vocab`.
///
/// ## Arguments
/// * `byte_vocab` - The byte vocabulary to validate against.
/// * `pairs` - The pair token map to validate.
///
/// ## Returns
/// A `Result` indicating whether the maps are compatible.
pub fn try_validate_pair_map<T: TokenType>(
    byte_vocab: &ByteMapVocab<T>,
    pairs: &PairTokenMap<T>,
) -> anyhow::Result<()> {
    let pair_targets: CommonHashSet<T> = pairs.values().copied().collect();

    for t in &pair_targets {
        if let Some(b) = byte_vocab.get_byte(*t) {
            anyhow::bail!("Target token in pair map {t:?} also mapped to byte {b:0x?}");
        }
    }

    for (&pair, &t) in pairs.iter() {
        for pt in [pair.0, pair.1] {
            let is_pair_target = pair_targets.contains(&pt);
            let byte_target = byte_vocab.get_byte(pt);

            if is_pair_target && let Some(b) = byte_target {
                anyhow::bail!(
                    "Pair {pair:?} -> {t:?} parent {pt:?} is a pair target and byte target: {b:0x?}"
                );
            }
            if !is_pair_target && byte_target.is_none() {
                anyhow::bail!("Pair {pair:?} -> {t:?} parent {pt:?} is not defined");
            }
        }
    }

    Ok(())
}

/// Pair ``(T, T) -> T`` Vocabulary.
///
/// - Grounded in a `ByteTable<T>` for byte-to-token mapping.
/// - Collection of ``(T, T) -> T`` pairs.
#[derive(Default, Debug, Clone)]
pub struct PairMapVocab<T: TokenType> {
    /// Byte/token mapping table.
    pub byte_vocab: ByteMapVocab<T>,

    /// Map of ``{ (T, T) -> T }``.
    pub pair_map: PairTokenMap<T>,
}

impl<T: TokenType> PairMapVocab<T> {
    /// Initialize a [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `pairs` - The pair token map.
    ///
    /// ## Returns
    /// A `Result` containing the new `PairMapVocab` instance or an error.
    pub fn new(
        byte_vocab: ByteMapVocab<T>,
        mut pairs: PairTokenMap<T>,
    ) -> anyhow::Result<Self> {
        try_validate_pair_map(&byte_vocab, &pairs)?;
        pairs.shrink_to_fit();
        Ok(Self {
            byte_vocab,
            pair_map: pairs,
        })
    }

    /// Get the map of pairs.
    pub fn pairs(&self) -> &PairTokenMap<T> {
        &self.pair_map
    }

    /// Get the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.byte_vocab.len() + self.pair_map.len()
    }

    /// Is this empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Looks up a pair.
    ///
    /// ## Arguments
    /// * `pair` - The pair of tokens to look up.
    ///
    /// ## Returns
    /// An `Option` containing the token corresponding to the pair if it exists.
    pub fn lookup_pair(
        &self,
        pair: &Pair<T>,
    ) -> Option<T> {
        self.pair_map.get(pair).copied()
    }
}

impl<T: TokenType> TokenVocab<T> for PairMapVocab<T> {
    fn unordered_tokens(&self) -> impl Iterator<Item = T> {
        self.byte_vocab
            .unordered_tokens()
            .chain(self.pair_map.values().copied())
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        let decoder = PairExpansionDecoder::from_pair_vocab(self);

        self.byte_vocab.span_pairs().chain(
            self.pair_map
                .values()
                .map(move |&t| (decoder.try_decode_to_bytes(&[t]).unwrap().unwrap(), t)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_sorted() {
        type T = u32;
        let byte_vocab: ByteMapVocab<T> = Default::default();

        let mut vocab = PairMapVocab::<T> {
            pair_map: PairTokenMap::default(),
            byte_vocab: byte_vocab.clone(),
        };

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(&vocab.sorted_tokens(), &byte_vocab.sorted_tokens());

        vocab.pair_map.insert((1, 2), 300);
        vocab.pair_map.insert((3, 4), 301);
        vocab.pair_map.insert((300, 301), 302);

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
}
