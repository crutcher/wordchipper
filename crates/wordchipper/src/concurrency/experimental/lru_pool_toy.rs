//! # Thread Pool Toy

use core::fmt::Debug;
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use parking_lot::lock_api::RwLock;

use crate::{
    concurrency::threads::{resolve_max_pool, unstable_current_thread_id_hash},
    types::CommonHashSet,
};

/// Experimental LRU-based variant of [`crate::concurrency::PoolToy`].
///
/// This appears to provide no benefit in benchmarks.
///
/// The idea is that in high-contention applications,
/// even one instance of collision could be costly;
/// so this variant would use an LRU cache of recently
/// used thread IDs to pool items.
///
/// Alternatively, by enabling `probe`, we can also
/// scan for recent hash allocations and attempt to
/// select an unused pool index.
pub struct LruPoolToy<T>
where
    T: Clone + Send,
{
    pool: Vec<T>,

    probe: bool,
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
    /// Initialize a new thread-local pool with the given item and maximum pool size.
    ///
    /// ## Arguments
    /// * `pool` - the pool of items.
    /// * `probe` - whether to use aggressive probing to avoid collisions.
    /// * `max_pool` - override the maximum pool size, see [`resolve_max_pool`].
    pub fn new(
        item: T,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        Self::new_with_probe(item, false, max_pool)
    }

    /// Initialize a new thread-local pool with the given item and maximum pool size.
    ///
    /// ## Arguments
    /// * `pool` - the pool of items.
    /// * `probe` - whether to use aggressive probing to avoid collisions.
    /// * `max_pool` - override the maximum pool size, see [`resolve_max_pool`].
    pub fn new_with_probe(
        item: T,
        probe: bool,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let size = resolve_max_pool(max_pool);
        Self::from_pool(vec![item; size], probe)
    }

    /// Create a new thread-local pool with the given vector of items.
    ///
    /// ## Arguments
    /// * `pool` - the pool of items.
    /// * `probe` - whether to use aggressive probing to avoid collisions.
    pub fn from_pool(
        pool: Vec<T>,
        probe: bool,
    ) -> Self {
        assert!(!pool.is_empty());
        let len = pool.len();
        Self {
            pool,
            probe,
            next: AtomicUsize::new(0),
            lru: RwLock::new(hashlru::Cache::new(len * 2)),
        }
    }

    /// Get a reference to the item for the current thread.
    pub fn get(&self) -> &T {
        let tid = unstable_current_thread_id_hash();

        let mut writer = self.lru.write();
        let idx: usize = match writer.get(&tid) {
            Some(idx) => *idx,
            None => {
                let mut idx = self
                    .next
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                        Some((x + 1) % self.pool.len())
                    })
                    .unwrap_or(0);

                // This also doesn't seem to help.
                if self.probe {
                    // Be even more aggressive about avoiding collisions.
                    let active = writer.values().copied().collect::<CommonHashSet<_>>();
                    for j in 0..self.pool.len() {
                        if !active.contains(&j) {
                            idx = j;
                        }
                    }
                }

                writer.insert(tid, idx);
                idx
            }
        };
        &self.pool[idx]
    }

    /// Get the length of the pool.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Is this empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Clone for LruPoolToy<T>
where
    T: Clone + Send,
{
    fn clone(&self) -> Self {
        Self::from_pool(self.pool.clone(), self.probe)
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
        let pool = LruPoolToy::new(10, max_pool);

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
