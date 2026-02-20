//! # Resource Loader

#[cfg(feature = "std")]
use std::path::PathBuf;

#[cfg(feature = "std")]
use crate::support::resources::KeyedResource;

/// A trait for loading resources.
pub trait ResourceLoader {
    /// Load a resource.
    #[cfg(feature = "std")]
    fn load_resource_path(
        &mut self,
        resource: &KeyedResource,
    ) -> crate::WCResult<PathBuf>;
}

#[cfg(feature = "download")]
impl ResourceLoader for crate::disk_cache::WordchipperDiskCache {
    #[cfg(feature = "std")]
    fn load_resource_path(
        &mut self,
        resource: &KeyedResource,
    ) -> crate::WCResult<PathBuf> {
        self.load_cached_path(&resource.key, &resource.resource.urls, true)
            .map_err(|e| crate::WCError::External(e.to_string()))
    }
}
