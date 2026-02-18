//! # Resource Loader

#[cfg(feature = "std")]
use std::path::PathBuf;

#[cfg(feature = "std")]
use crate::resources::KeyedResource;

/// A trait for loading resources.
pub trait ResourceLoader {
    /// Load a resource.
    #[cfg(feature = "std")]
    fn load_resource_path(
        &mut self,
        resource: &KeyedResource,
    ) -> anyhow::Result<PathBuf>;
}

#[cfg(feature = "download")]
impl ResourceLoader for crate::disk_cache::WordchipperDiskCache {
    #[cfg(feature = "std")]
    fn load_resource_path(
        &mut self,
        resource: &KeyedResource,
    ) -> anyhow::Result<PathBuf> {
        self.load_cached_path(&resource.key, &resource.resource.urls, true)
    }
}
