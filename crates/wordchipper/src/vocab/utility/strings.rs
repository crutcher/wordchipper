//! # String Utilities

use crate::alloc::borrow::Cow;
use crate::alloc::string::String;
use crate::alloc::vec::Vec;

/// "stable" stub for for [`String::from_utf8_lossy`].
pub fn string_from_lossy_utf8(v: Vec<u8>) -> String {
    if let Cow::Owned(string) = String::from_utf8_lossy(&v) {
        string
    } else {
        // SAFETY: `String::from_utf8_lossy`'s contract ensures that if
        // it returns a `Cow::Borrowed`, it is a valid UTF-8 string.
        // Otherwise, it returns a new allocation of an owned `String`, with
        // replacement characters for invalid sequences, which is returned
        // above.
        unsafe { String::from_utf8_unchecked(v) }
    }
}
