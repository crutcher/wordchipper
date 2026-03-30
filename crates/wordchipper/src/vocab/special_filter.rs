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
