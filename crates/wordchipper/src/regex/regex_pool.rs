//! # Thread Regex Pool
#![allow(unused)]

use crate::alloc::sync::Arc;
use crate::regex::regex_supplier::RegexSupplier;
use crate::regex::regex_wrapper::RegexWrapper;
use crate::types::CommonHashMap;
use core::fmt::Debug;
use core::num::NonZero;
use parking_lot::RwLock;
use std::thread::ThreadId;

fn unsafe_threadid_to_u64(thread_id: &ThreadId) -> u64 {
    unsafe { std::mem::transmute(thread_id) }
}

/// Interior-Mutable Thread-Local Regex Pool
///
/// In HPC applications, under some loads, interior buffers in compiled regex
/// can block. This pool exists to mitigate that, by cloning regex-per-thread.
#[derive(Clone)]
pub struct RegexWrapperPool {
    regex: Arc<RegexWrapper>,

    max_pool: u64,
    pool: Arc<RwLock<CommonHashMap<u64, Arc<RegexWrapper>>>>,
}

impl Debug for RegexWrapperPool {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("RegexPool")
            .field("regex", &self.regex)
            .finish()
    }
}

impl From<Arc<RegexWrapper>> for RegexWrapperPool {
    fn from(regex: Arc<RegexWrapper>) -> Self {
        Self::new(regex)
    }
}

impl RegexWrapperPool {
    /// Create a new `RegexPool`
    ///
    /// ## Arguments
    /// * `regex` - The regex to pool.
    ///
    /// ## Returns
    /// A new `RegexWrapperPool` instance.
    pub fn new(regex: Arc<RegexWrapper>) -> Self {
        let max_pool = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(128).unwrap())
            .get() as u64;

        Self {
            regex,
            max_pool,
            pool: Arc::new(RwLock::new(Default::default())),
        }
    }

    /// Returns the number of regex instances in the pool.
    pub fn len(&self) -> usize {
        self.pool.read().len()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.pool.read().is_empty()
    }

    /// Clear the internal regex pool.
    pub fn clear(&self) {
        self.pool.write().clear();
    }
}

impl RegexSupplier for RegexWrapperPool {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        let thread_id = std::thread::current().id();
        let slot = unsafe_threadid_to_u64(&thread_id) % self.max_pool;

        if let Some(regex) = self.pool.read().get(&slot) {
            return regex.clone();
        }

        let mut writer = self.pool.write();
        let re = Arc::new((*self.regex).clone());
        writer.insert(slot, re.clone());
        re
    }

    fn get_pattern(&self) -> String {
        self.regex.as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use crate::regex::regex_wrapper::RegexWrapperPattern;

    #[test]
    fn test_regex_pool() {
        let pattern: RegexWrapperPattern = r"foo".into();
        let regex: Arc<RegexWrapper> = pattern.compile().unwrap().into();

        let pool: RegexWrapperPool = regex.clone().into();
        assert_eq!(pool.get_pattern(), r"foo");
        assert!(format!("{:?}", pool).contains(&format!("{:?}", regex).to_string()));

        let r0 = pool.get_regex();
        assert_eq!(r0.as_str(), r"foo");

        assert!(Arc::ptr_eq(&r0, &pool.get_regex()));

        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());

        pool.clear();

        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }
}
