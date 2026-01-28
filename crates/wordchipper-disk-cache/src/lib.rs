//! # wordchipper-disk-cache
#![warn(missing_docs)]

use crate::path_resolver::PathResolver;

pub mod disk_cache;
pub mod path_resolver;
pub mod path_utils;

pub use disk_cache::{WordchipperDiskCache, WordchipperDiskCacheOptions};

/// Environment variable key to override the default cache directory.
pub const WORDCHIPPER_CACHE_DIR: &str = "WORDCHIPPER_CACHE_DIR";
/// Environment variable key to override the default data directory.
pub const WORDCHIPPER_DATA_DIR: &str = "WORDCHIPPER_DATA_DIR";

/// Default [`PathResolver`] for wordchipper.
pub const WORDCHIPPER_CACHE_CONFIG: PathResolver = PathResolver {
    qualifier: "io.crates.wordchipper",
    organization: "",
    application: "wordchipper",
    cache_env_vars: &[WORDCHIPPER_CACHE_DIR],
    data_env_vars: &[WORDCHIPPER_DATA_DIR],
};
