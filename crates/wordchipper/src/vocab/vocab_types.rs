//! # Vocabulary Types

use crate::alloc::vec::Vec;
use crate::types::{CommonHashMap, Pair};

/// [`Pair<T>`] to `T` map.
pub type PairTokenMap<T> = CommonHashMap<Pair<T>, T>;

/// `T` to [`Pair<T>`] map.
pub type TokenPairMap<T> = CommonHashMap<T, Pair<T>>;

/// `Vec<u8>` to `T` map.
pub type SpanTokenMap<T> = CommonHashMap<Vec<u8>, T>;

/// `T` to `Vec<u8>` map.
pub type TokenSpanMap<T> = CommonHashMap<T, Vec<u8>>;

/// `T` to `u8` map.
pub type TokenByteMap<T> = CommonHashMap<T, u8>;

/// `u8` to `T` map.
pub type ByteTokenMap<T> = CommonHashMap<u8, T>;

/// `u8` to `T` array.
pub type ByteTokenArray<T> = [T; 256];
