//! # Common Types and Traits
use core::{
    fmt::{Debug, Display},
    hash::Hash,
};

use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Unsigned};

/// A type that can be used as a token in a BPE-based encoders.
///
/// These are constrained to be unsigned primitive integers;
/// such that the max token in a vocabulary is less than `T::max()`.
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

cfg_if::cfg_if! {
    if #[cfg(feature = "ahash")] {
        /// Type Alias for hash maps in this crate.
        pub type CommonHashMap<K, V> = ahash::AHashMap<K, V>;

        /// Iterator over hash map entries.
        ///
        /// Note: `ahash::AHashMap` is a specialization of `std::collections::HashMap`.
        pub type CommonHashIter<'a, K, V> = std::collections::hash_map::Iter<'a, K, V>;

        /// Type Alias for hash sets in this crate.
        pub type CommonHashSet<V> = ahash::AHashSet<V>;
    } else if #[cfg(feature = "foldhash")] {
        /// Type Alias for hash maps in this crate.
        pub type CommonHashMap<K, V> = foldhash::HashMap<K, V>;

        /// Iterator over hash map entries.
        ///
        /// Note: `foldhash::HashMap` is a specialization of `std::collections::HashMap`.
        pub type CommonHashIter<'a, K, V> = std::collections::hash_map::Iter<'a, K, V>;

        /// Type Alias for hash sets in this crate.
        pub type CommonHashSet<V> = foldhash::HashSet<V>;
    } else if #[cfg(feature = "std")] {
        /// Type Alias for hash maps in this crate.
        pub type CommonHashMap<K, V> = std::collections::HashMap<K, V>;

        /// Iterator over hash map entries.
        pub type CommonHashIter<'a, K, V> = std::collections::hash_map::Iter<'a, K, V>;

        /// Type Alias for hash sets in this crate.
        pub type CommonHashSet<V> = std::collections::HashSet<V>;
    } else if #[cfg(feature = "no_std")] {
        /// Type Alias for hash maps in this crate.
        pub type CommonHashMap<K, V> = hashbrown::HashMap<K, V>;

        /// Iterator over hash map entries.
        pub type CommonHashIter<'a, K, V> = hashbrown::hash_map::Iter<'a, K, V>;

        /// Type Alias for hash sets in this crate.
        pub type CommonHashSet<V> = hashbrown::HashSet<V>;
    } else {
        /// This error exists to give users more direct feedback
        /// on the feature configuration over the other compilation
        /// errors they would encounter from lacking the types.
        compile_error!("not(\"std\") requires \"no_std\" feature");
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use super::*;

    #[test]
    fn test_common_token_types() {
        struct IsToken<T: TokenType>(PhantomData<T>);

        let _: IsToken<u16>;
        let _: IsToken<u32>;
        let _: IsToken<u64>;
        let _: IsToken<usize>;
    }
}
