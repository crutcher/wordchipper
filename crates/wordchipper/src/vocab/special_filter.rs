use crate::{
    WCHashSet,
    prelude::*,
};

/// A policy for filtering special tokens.
#[derive(Default, Debug, Clone, PartialEq)]
pub enum SpecialFilter {
    /// Include all special tokens.
    #[default]
    IncludeAll,

    /// Exclude all special tokens.
    IncludeNone,

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
            SpecialFilter::IncludeAll => true,
            SpecialFilter::IncludeNone => false,
            SpecialFilter::Include(set) => set.contains(token),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_include_all() {
        let filter = SpecialFilter::IncludeAll;
        assert!(filter.contains("foo"));
    }

    #[test]
    fn test_include_none() {
        let filter = SpecialFilter::IncludeNone;
        assert!(!filter.contains("foo"));
    }

    #[test]
    fn test_include_some() {
        let filter = SpecialFilter::Include(["foo", "bar"].iter().map(|s| s.to_string()).collect());
        assert!(filter.contains("foo"));
        assert!(filter.contains("bar"));
        assert!(!filter.contains("baz"));
    }
}
