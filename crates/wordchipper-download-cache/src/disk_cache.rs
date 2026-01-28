//! # Wordchipper Disk Cache

use crate::WORDCHIPPER_CACHE_CONFIG;
use anyhow::Context;
use std::path::{Path, PathBuf};

/// Options for [`DiskDownloadCache`].
#[derive(Clone, Default, Debug)]
pub struct DiskDownloadCacheOptions {
    /// Optional path to the cache directory.
    ///
    pub cache_dir: Option<PathBuf>,

    /// Optional path to the data directory.
    pub data_dir: Option<PathBuf>,
}

/// Disk cache for downloaded files.
#[derive(Clone, Debug)]
pub struct DiskDownloadCache {
    /// Cache directory.
    pub cache_dir: PathBuf,

    /// Data directory.
    pub data_dir: PathBuf,
}

/// Extend a path with a context and filename.
///
/// * Does not check that the path exists.
/// * Does not initialize the containing directories.
///
/// # Arguments
/// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
/// * `file` - the final file name.
pub fn extend_path<P, S, F>(
    path: P,
    context: &[S],
    filename: F,
) -> PathBuf
where
    P: AsRef<Path>,
    S: AsRef<str>,
    F: AsRef<str>,
{
    let mut path = path.as_ref().to_path_buf();
    path.extend(context.iter().map(|s| s.as_ref()));
    path.push(filename.as_ref());
    path
}

impl DiskDownloadCache {
    /// Construct a new [`DiskDownloadCache`].
    pub fn init(options: DiskDownloadCacheOptions) -> anyhow::Result<Self> {
        let cache_dir = WORDCHIPPER_CACHE_CONFIG
            .resolve_cache_dir(options.cache_dir)
            .context("failed to resolve cache directory")?;

        let data_dir = WORDCHIPPER_CACHE_CONFIG
            .resolve_data_dir(options.data_dir)
            .context("failed to resolve data directory")?;

        Ok(Self {
            cache_dir,
            data_dir,
        })
    }

    /// Get the cache path for the given key.
    ///
    /// * Does not check that the path exists.
    /// * Does not initialize the containing directories.
    ///
    /// # Arguments
    /// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
    /// * `file` - the final file name.
    pub fn cache_path<C, F>(
        &self,
        context: &[C],
        file: F,
    ) -> PathBuf
    where
        C: AsRef<str>,
        F: AsRef<str>,
    {
        extend_path(&self.cache_dir, context, file)
    }

    /// Get the data path for the given key.
    ///
    /// * Does not check that the path exists.
    /// * Does not initialize the containing directories.
    ///
    /// # Arguments
    /// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
    /// * `file` - the final file name.
    pub fn data_path<C, F>(
        &self,
        context: &[C],
        file: F,
    ) -> PathBuf
    where
        C: AsRef<str>,
        F: AsRef<str>,
    {
        extend_path(&self.data_dir, context, file)
    }
}

#[cfg(test)]
mod tests {
    use crate::disk_cache::{DiskDownloadCache, DiskDownloadCacheOptions};
    use crate::{WORDCHIPPER_CACHE_CONFIG, WORDCHIPPER_CACHE_DIR, WORDCHIPPER_DATA_DIR};
    use serial_test::serial;
    use std::env;
    use std::path::PathBuf;

    #[test]
    #[serial]
    fn test_resolve_dirs() {
        let orig_cache_dir = env::var(WORDCHIPPER_CACHE_DIR);
        let orig_data_dir = env::var(WORDCHIPPER_CACHE_DIR);

        let pds = WORDCHIPPER_CACHE_CONFIG
            .project_dirs()
            .expect("failed to get project dirs");

        let user_cache_dir = PathBuf::from("/tmp/wordchipper/cache");
        let user_data_dir = PathBuf::from("/tmp/wordchipper/data");

        let env_cache_dir = PathBuf::from("/tmp/wordchipper/env_cache");
        let env_data_dir = PathBuf::from("/tmp/wordchipper/env_data");

        // No env vars
        unsafe {
            env::remove_var(WORDCHIPPER_CACHE_DIR);
            env::remove_var(WORDCHIPPER_DATA_DIR);
        }

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions {
            cache_dir: Some(user_cache_dir.clone()),
            data_dir: Some(user_data_dir.clone()),
        })
        .unwrap();
        assert_eq!(&cache.cache_dir, &user_cache_dir);
        assert_eq!(&cache.data_dir, &user_data_dir);

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir, &pds.cache_dir().to_path_buf());
        assert_eq!(&cache.data_dir, &pds.data_dir().to_path_buf());

        // With env var.
        unsafe {
            env::set_var(WORDCHIPPER_CACHE_DIR, env_cache_dir.to_str().unwrap());
            env::set_var(WORDCHIPPER_DATA_DIR, env_data_dir.to_str().unwrap());
        }

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions {
            cache_dir: Some(user_cache_dir.clone()),
            data_dir: Some(user_data_dir.clone()),
        })
        .unwrap();
        assert_eq!(&cache.cache_dir, &user_cache_dir);
        assert_eq!(&cache.data_dir, &user_data_dir);

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir, &env_cache_dir);
        assert_eq!(&cache.data_dir, &env_data_dir);

        // restore original env var.
        match orig_cache_dir {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_CACHE_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_CACHE_DIR) },
        }
        match orig_data_dir {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_DATA_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_DATA_DIR) },
        }
    }

    #[test]
    fn test_data_path() {
        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions::default()).unwrap();
        let path = cache.data_path(&["prefix"], "file.txt");
        assert_eq!(path, cache.data_dir.join("prefix").join("file.txt"));
    }

    #[test]
    fn test_cache_path() {
        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions::default()).unwrap();
        let path = cache.cache_path(&["prefix"], "file.txt");
        assert_eq!(path, cache.cache_dir.join("prefix").join("file.txt"));
    }
}
