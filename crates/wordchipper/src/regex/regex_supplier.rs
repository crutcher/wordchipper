//! # Regex Supplier Trait

use crate::alloc::fmt::Debug;
use crate::alloc::string::String;
use crate::alloc::string::ToString;
use crate::alloc::sync::Arc;
use crate::regex::{RegexWrapper, RegexWrapperHandle};

/// Common Regex Supplier Handle Type
pub type RegexSupplierHandle = Arc<dyn RegexSupplier>;

/// Regex Supplier Trait
pub trait RegexSupplier: Sync + Send + Debug {
    /// Get the regex.
    fn get_regex(&self) -> Arc<RegexWrapper>;

    /// Get the regex pattern.
    fn get_pattern(&self) -> String {
        self.get_regex().as_str().to_string()
    }
}

impl RegexSupplier for RegexWrapper {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        Arc::new(self.clone())
    }
}

impl RegexSupplier for RegexWrapperHandle {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex::{RegexWrapperHandle, RegexWrapperPattern};

    #[test]
    fn test_regex_supplier() {
        let pattern: RegexWrapperPattern = r"foo".into();
        let wrapper: RegexWrapper = pattern.compile().unwrap();
        assert_eq!(wrapper.get_regex().as_ref(), &wrapper);

        let wrapper_handle: RegexWrapperHandle = wrapper.clone().into();
        assert_eq!(wrapper_handle.get_regex().as_ref(), &wrapper);

        let supplier: RegexSupplierHandle = wrapper_handle;
        assert_eq!(supplier.get_regex().as_ref(), &wrapper);

        assert_eq!(supplier.get_pattern(), "foo");
        assert_eq!(supplier.get_regex().as_str(), "foo");
    }
}
