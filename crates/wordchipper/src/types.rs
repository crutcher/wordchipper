//! # Common Types and Traits
use core::fmt::{Debug, Display};
use core::hash::Hash;
use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Unsigned};

/// A type that can be used as a token in a BPE-based encoders.
pub trait TokenType:
    'static
    + PrimInt
    + FromPrimitive
    + ToPrimitive
    + Unsigned
    + Hash
    + Default
    + Debug
    + Display
    + Send
    + Sync
{
}

impl<T> TokenType for T where
    T: 'static
        + PrimInt
        + FromPrimitive
        + ToPrimitive
        + Unsigned
        + Hash
        + Default
        + Debug
        + Display
        + Send
        + Sync
{
}

/// A pair of tokens.
pub type Pair<T> = (T, T);

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

/// Check if a type is `Send`.
#[cfg(test)]
pub(crate) fn check_is_send<S: Send>(_: S) {}

#[cfg(test)]
/// Check if a type is `Sync`.
pub(crate) fn check_is_sync<S: Sync>(_: S) {}

#[cfg(test)]
mod tests {
    use super::*;
    use core::marker::PhantomData;

    #[test]
    fn test_common_token_types() {
        struct IsToken<T: TokenType>(PhantomData<T>);

        let _: IsToken<u16>;
        let _: IsToken<u32>;
        let _: IsToken<u64>;
        let _: IsToken<usize>;
    }
}
