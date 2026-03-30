use crate::{
    WCHashSet,
    prelude::*,
};

/// A policy for filtering special tokens.
#[derive(Default, Debug, Clone, PartialEq)]
pub enum SpecialFilter {
    /// Include all special tokens.
    #[default]
    All,

    /// Exclude all special tokens.
    None,

    /// Only include the specified special tokens.
    Include(WCHashSet<String>),
}

impl SpecialFilter {
    /// Does the filter permit the given special token?
    pub fn contains(
        &self,
        token: &str,
    ) -> bool {
        match self {
            SpecialFilter::All => true,
            SpecialFilter::None => false,
            SpecialFilter::Include(set) => set.contains(token),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_all() {
        let filter = SpecialFilter::All;
        assert!(filter.contains("foo"));
    }

    #[test]
    fn test_special_none() {
        let filter = SpecialFilter::None;
        assert!(!filter.contains("foo"));
    }

    #[test]
    fn test_special_include() {
        let filter = SpecialFilter::Include(["foo", "bar"].iter().map(|s| s.to_string()).collect());
        assert!(filter.contains("foo"));
        assert!(filter.contains("bar"));
        assert!(!filter.contains("baz"));
    }
}
