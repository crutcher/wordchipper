//! # Thread Pool Toy

use crate::concurrency::threads;
use crate::concurrency::threads::resolve_max_pool;
use core::fmt::Debug;
use parking_lot::lock_api::RwLock;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lru Cached Thread -> T Pool.
pub struct LruPoolToy<T>
where
    T: Clone + Send,
{
    pool: Vec<T>,

    next: AtomicUsize,
    lru: parking_lot::RwLock<hashlru::Cache<usize, usize>>,
}

impl<T> AsRef<T> for LruPoolToy<T>
where
    T: Clone + Send,
{
    fn as_ref(&self) -> &T {
        self.get()
    }
}

impl<T> LruPoolToy<T>
where
    T: Clone + Send,
{
    /// Create a new thread-local pool with the given vector of items.
    pub fn new(pool: Vec<T>) -> Self {
        assert!(!pool.is_empty());
        let len = pool.len();
        Self {
            pool,
            next: AtomicUsize::new(0),
            lru: RwLock::new(hashlru::Cache::new(len)),
        }
    }

    /// Initialize a new thread-local pool with the given item and maximum pool size.
    pub fn init(
        item: T,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let size = resolve_max_pool(max_pool);
        Self::new(vec![item; size])
    }

    /// Get a reference to the item for the current thread.
    pub fn get(&self) -> &T {
        let tid = threads::unstable_current_thread_id_hash();

        let mut writer = self.lru.write();
        let idx: usize = match writer.get(&tid) {
            Some(idx) => *idx,
            None => {
                let idx = self
                    .next
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                        Some((x + 1) % self.pool.len())
                    })
                    .unwrap_or(0);
                writer.insert(tid, idx);
                idx
            }
        };
        &self.pool[idx]
    }

    /// Get the length of the pool.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.pool.len()
    }
}

impl<T> Clone for LruPoolToy<T>
where
    T: Clone + Send,
{
    fn clone(&self) -> Self {
        Self::new(self.pool.clone())
    }
}

impl<T> Debug for LruPoolToy<T>
where
    T: Clone + Send + Debug,
{
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("PoolToy")
            .field("item", &self.pool[0])
            .field("len", &self.pool.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concurrency::threads::resolve_max_pool;

    #[test]
    fn test_pool_toy() {
        let max_pool = Some(NonZeroUsize::new(128).unwrap());
        let pool = LruPoolToy::init(10, max_pool);

        // This will be different sizes on different systems.
        let size = resolve_max_pool(max_pool);

        assert_eq!(pool.len(), size);
        assert_eq!(&pool.pool, vec![10; size].as_slice());

        assert_eq!(pool.get(), &10);
        assert_eq!(pool.as_ref(), &10);

        assert_eq!(
            format!("{:?}", pool),
            format!("PoolToy {{ item: 10, len: {size} }}")
        );

        let clone = pool.clone();
        assert_eq!(&clone.pool, &pool.pool);
    }
}
