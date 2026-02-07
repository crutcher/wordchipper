//! # Vocabulary Types

use crate::alloc::vec::Vec;
use crate::types::{CommonHashMap, Pair};

/// `{ Pair<T> -> T}` map.
///
/// ## Style Hints
/// Instance names should prefer `pair_map`, or `pair_token_map`.
pub type PairTokenMap<T> = CommonHashMap<Pair<T>, T>;

/// `{ T -> Pair<T> }` map.
///
/// ## Style Hints
/// Instance names should prefer `taken_pairs`, or `token_pair_map`.
pub type TokenPairMap<T> = CommonHashMap<T, Pair<T>>;

/// `{ Vec<u8> -> T }` map.
///
/// ## Style Hints
/// Instance names should prefer `span_map`, or `span_token_map`.
pub type SpanTokenMap<T> = CommonHashMap<Vec<u8>, T>;

/// `{ T -> Vec<u8> }` map.
///
/// ## Style Hints
/// Instance names should prefer `token_spans`, or `token_span_map`.
pub type TokenSpanMap<T> = CommonHashMap<T, Vec<u8>>;

/// `{ T -> u8 }` map.
///
/// ## Style Hints
/// Instance names should prefer `token_bytes`, or `token_byte_map`.
pub type TokenByteMap<T> = CommonHashMap<T, u8>;

/// `{ u8 -> T }` map.
///
/// ## Style Hints
/// Instance names should prefer `byte_tokens`, or `byte_token_map`.
pub type ByteTokenMap<T> = CommonHashMap<u8, T>;

/// `[T; 256]` array.
///
/// ## Style Hints
/// Instance names should prefer `byte_tokens`, or `byte_token_array`.
pub type ByteTokenArray<T> = [T; 256];
