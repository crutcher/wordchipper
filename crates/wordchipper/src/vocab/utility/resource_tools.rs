//! # Remote Resource Tools
/// A resource with a constant URL and optional hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstUrlResource {
    /// The URL associated with this resource.
    pub url: &'static str,

    /// The hash associated with this resource, if available.
    pub hash: Option<&'static str>,
}

impl ConstUrlResource {
    /// Create a new [`ConstUrlResource`].
    pub const fn new(
        url: &'static str,
        hash: Option<&'static str>,
    ) -> Self {
        Self { url, hash }
    }

    /// Create a new [`ConstUrlResource`] with no hash.
    pub const fn no_hash(url: &'static str) -> Self {
        Self::new(url, None)
    }
}
