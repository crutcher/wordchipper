//! # Remote Resource Tools
/// A resource with a constant URL and optional hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstUrlResource {
    /// The URL associated with this resource.
    pub urls: &'static [&'static str],

    /// The hash associated with this resource, if available.
    pub hash: Option<&'static str>,
}
