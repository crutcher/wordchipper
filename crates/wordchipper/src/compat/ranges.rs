//! # Range Utilities

use core::ops::Range;

use num_traits::PrimInt;

/// Add an offset to the start and end of a [`Range<Idx>`].
pub fn offset_range<Idx>(
    range: Range<usize>,
    offset: usize,
) -> Range<usize>
where
    Idx: PrimInt,
{
    Range {
        start: range.start + offset,
        end: range.end + offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_range() {
        assert_eq!(offset_range::<usize>(0..10, 5), 5..15);
    }
}
