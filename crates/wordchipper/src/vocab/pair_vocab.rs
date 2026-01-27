//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::alloc::sync::Arc;
use crate::alloc::vec::Vec;
use crate::decoders::TokenDecoder;
use crate::decoders::utility::pair_decoder::PairExpansionDecoder;
use crate::types::{CommonHashSet, Pair, PairTokenMap, TokenType};
use crate::vocab::byte_vocab::ByteMapVocab;
use crate::vocab::token_vocab::TokenVocab;

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
    byte_vocab: Arc<ByteMapVocab<T>>,

    /// Map of ``{ (T, T) -> T }``.
    pairs: PairTokenMap<T>,
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
    pub fn init<B>(
        byte_vocab: B,
        pairs: PairTokenMap<T>,
    ) -> anyhow::Result<Self>
    where
        B: Into<Arc<ByteMapVocab<T>>>,
    {
        let byte_vocab = byte_vocab.into();
        try_validate_pair_map(&byte_vocab, &pairs)?;
        Ok(Self { byte_vocab, pairs })
    }

    /// Get the byte/token mapping table.
    ///
    /// ## Returns
    /// A reference to the internal `ByteMapVocab` arc.
    pub fn byte_vocab(&self) -> &Arc<ByteMapVocab<T>> {
        &self.byte_vocab
    }

    /// Get the map of pairs.
    ///
    /// ## Returns
    /// A reference to the internal `PairTokenMap`.
    pub fn pairs(&self) -> &PairTokenMap<T> {
        &self.pairs
    }

    /// Get the number of tokens in the vocabulary.
    ///
    /// ## Returns
    /// The total number of tokens (bytes + pairs).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.byte_vocab.len() + self.pairs.len()
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
    ) -> Option<&T> {
        self.pairs.get(pair)
    }
}

impl<T: TokenType> TokenVocab<T> for PairMapVocab<T> {
    fn unordered_tokens(&self) -> impl Iterator<Item = T> {
        self.byte_vocab
            .unordered_tokens()
            .chain(self.pairs.values().copied())
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        let decoder = PairExpansionDecoder::from_pair_vocab(self);

        self.byte_vocab.span_pairs().chain(
            self.pairs
                .values()
                .map(move |&t| (decoder.try_decode_to_bytes([t]).unwrap(), t)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::sync::Arc;

    #[test]
    fn test_tokens_sorted() {
        type T = u32;
        let byte_vocab: Arc<ByteMapVocab<T>> = Arc::new(Default::default());

        let mut vocab = PairMapVocab::<T> {
            pairs: PairTokenMap::default(),
            byte_vocab: byte_vocab.clone(),
        };

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(&vocab.sorted_tokens(), &byte_vocab.sorted_tokens());

        vocab.pairs.insert((1, 2), 300);
        vocab.pairs.insert((3, 4), 301);
        vocab.pairs.insert((300, 301), 302);

        assert_eq!(vocab.max_token(), 302);

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
