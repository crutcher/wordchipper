//! # Regex Supplier Trait

use crate::alloc::fmt::Debug;
use crate::alloc::string::String;
use crate::alloc::string::ToString;
use crate::alloc::sync::Arc;
use crate::regex::RegexWrapper;

/// Common Regex Supplier Handle Type
pub type RegexSupplierHandle = Arc<dyn RegexSupplier>;

/// Regex Supplier Trait
pub trait RegexSupplier: Sync + Send {
    /// Get the regex.
    ///
    /// ## Returns
    /// An `Arc` containing the `RegexWrapper`.
    fn get_regex(&self) -> Arc<RegexWrapper>;

    /// Get the regex pattern.
    ///
    /// ## Returns
    /// The regex pattern as a `String`.
    fn get_pattern(&self) -> String {
        self.get_regex().as_str().to_string()
    }
}

impl Debug for dyn RegexSupplier {
    fn fmt(
        &self,
        f: &mut alloc::fmt::Formatter<'_>,
    ) -> alloc::fmt::Result {
        write!(f, "RegexSupplier({})", self.get_pattern())
    }
}

impl RegexSupplier for RegexWrapper {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        Arc::new(self.clone())
    }
}

impl RegexSupplier for Arc<RegexWrapper> {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        self.clone()
    }
}
