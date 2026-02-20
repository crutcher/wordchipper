//! # Slice Utilities

use crate::alloc::vec::Vec;

/// Converts a `&[Vec<T>]` into a `&[&[T]]`.
pub fn inner_slice_view<T, C: AsRef<[T]>>(x: &[C]) -> Vec<&[T]> {
    x.iter().map(|c| c.as_ref()).collect()
}

/// Converts a `&[String]` into a `&[&str]`.
pub fn inner_str_view<S: AsRef<str>>(x: &[S]) -> Vec<&str> {
    x.iter().map(|s| s.as_ref()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::vec;

    #[test]
    fn test_inner_slice_view() {
        let vecs = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let view = inner_slice_view(&vecs);

        assert_eq!(view, vec![&[1, 2, 3], &[4, 5, 6]]);

        for i in 0..view.len() {
            assert!(core::ptr::eq(view[i].as_ptr(), vecs[i].as_ptr()))
        }
    }

    #[test]
    fn test_inner_str_view() {
        let strings = vec!["hello", "world"];
        let view = inner_str_view(&strings);

        assert_eq!(view, vec!["hello", "world"]);
        for i in 0..view.len() {
            assert!(core::ptr::eq(view[i].as_ptr(), strings[i].as_ptr()))
        }
    }
}
