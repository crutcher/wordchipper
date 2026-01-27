//! # Common Types and Traits
use crate::alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::ops::{AddAssign, SubAssign};
use num_traits::bounds::UpperBounded;
use num_traits::{FromPrimitive, Num, ToPrimitive, Unsigned};

/// A type that can be used as a token in a BPE-based encoders.
pub trait TokenType:
    'static
    + Default
    + Debug
    + Clone
    + Copy
    + Hash
    + Send
    + Sync
    + Unsigned
    + FromPrimitive
    + ToPrimitive
    + UpperBounded
    + Ord
{
}

impl<T> TokenType for T where
    T: 'static
        + Default
        + Debug
        + Clone
        + Copy
        + Hash
        + Send
        + Sync
        + Unsigned
        + FromPrimitive
        + ToPrimitive
        + UpperBounded
        + Ord
{
}

/// A pair of tokens.
pub type Pair<T> = (T, T);

/// A type that can be used as a word count.
pub trait CountType:
    Num
    + AddAssign
    + SubAssign
    + Default
    + Copy
    + Debug
    + Display
    + Send
    + Sync
    + Hash
    + Ord
    + FromPrimitive
{
}

impl<T> CountType for T where
    T: Num
        + AddAssign
        + SubAssign
        + Default
        + Copy
        + Debug
        + Display
        + Send
        + Sync
        + Hash
        + Ord
        + FromPrimitive
{
}

/// A type that can be used as a string key.
pub trait StringChunkType:
    for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

impl<T> StringChunkType for T where
    T: for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

#[cfg(feature = "ahash")]
mod hash_types {
    /// Type Alias for hash maps in this crate.
    pub type CommonHashMap<K, V> = ahash::AHashMap<K, V>;

    /// Iterator over hash map entries.
    pub type CommonHashIter<'a, K, V> = std::collections::hash_map::Iter<'a, K, V>;

    /// Type Alias for hash sets in this crate.
    pub type CommonHashSet<V> = ahash::AHashSet<V>;
}
#[cfg(all(feature = "std", not(feature = "ahash")))]
mod hash_types {
    /// Type Alias for hash maps in this crate.
    pub type CommonHashMap<K, V> = std::collections::HashMap<K, V>;

    /// Iterator over hash map entries.
    pub type CommonHashIter<'a, K, V> = std::collections::hash_map::Iter<'a, K, V>;

    /// Type Alias for hash sets in this crate.
    pub type CommonHashSet<V> = std::collections::HashSet<V>;
}
#[cfg(all(not(feature = "std"), feature = "no_std"))]
mod hash_types {
    /// Type Alias for hash maps in this crate.
    pub type CommonHashMap<K, V> = hashbrown::HashMap<K, V>;

    /// Iterator over hash map entries.
    pub type CommonHashIter<'a, K, V> = hashbrown::hash_map::Iter<'a, K, V>;

    /// Type Alias for hash sets in this crate.
    pub type CommonHashSet<V> = hashbrown::HashSet<V>;
}
pub use hash_types::*;

/// [`Pair<T>`] to T map.
pub type PairTokenMap<T> = CommonHashMap<Pair<T>, T>;

/// T to [`Pair<T>`] map.
pub type TokenToPairMap<T> = CommonHashMap<T, Pair<T>>;

/// Byte vector to T map.
pub type SpanTokenMap<T> = CommonHashMap<Vec<u8>, T>;

/// T to byte vector map.
pub type TokenToWordMap<T> = CommonHashMap<T, Vec<u8>>;

/// Check if a type is `Send`.
#[cfg(test)]
pub(crate) fn check_is_send<S: Send>(_: S) {}

#[cfg(test)]
/// Check if a type is `Sync`.
pub(crate) fn check_is_sync<S: Sync>(_: S) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::String;
    use compact_str::CompactString;
    use core::marker::PhantomData;

    #[test]
    fn test_common_token_types() {
        struct IsToken<T: TokenType>(PhantomData<T>);

        let _: IsToken<u16>;
        let _: IsToken<u32>;
        let _: IsToken<u64>;
        let _: IsToken<usize>;
    }

    #[test]
    fn test_common_count_types() {
        struct IsCount<T: CountType>(PhantomData<T>);

        let _: IsCount<u16>;
        let _: IsCount<u32>;
        let _: IsCount<u64>;
        let _: IsCount<usize>;
    }

    #[test]
    fn test_common_string_chunk_types() {
        struct IsStringChunk<T: StringChunkType>(PhantomData<T>);

        let _: IsStringChunk<String>;
        let _: IsStringChunk<CompactString>;
    }
}
