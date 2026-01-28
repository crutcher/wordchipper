//! # wordchipper-download-cache
#![warn(missing_docs)]

use crate::path_resolver::PathResolver;

pub mod disk_cache;
pub mod path_resolver;

/// Environment variable key to override the default cache directory.
pub const WORDCHIPPER_CACHE_DIR: &str = "WORDCHIPPER_CACHE_DIR";
/// Environment variable key to override the default data directory.
pub const WORDCHIPPER_DATA_DIR: &str = "WORDCHIPPER_DATA_DIR";

/// Default [`PathResolver`] for wordchipper.
pub const WORDCHIPPER_CACHE_CONFIG: PathResolver = PathResolver {
    qualifier: "io",
    organization: "crates",
    application: "wordchipper",
    cache_env_vars: &[WORDCHIPPER_CACHE_DIR],
    data_env_vars: &[WORDCHIPPER_DATA_DIR],
};
