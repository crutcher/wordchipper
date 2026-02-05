//! # Thread Pool Toy

use crate::concurrency::threads;
use core::fmt::Debug;
use std::num::NonZeroUsize;

/// Current Thread -> T Pool.
///
/// This struct provides a thread-id hashed pool of items.
/// Rather than pure thread-local storage, the pool is
/// initialized with a vector of items, and the current
/// thread's ID is hashed to get the item.
///
/// ## Style Hints for AI
///
/// Instance names should prefer `${T-name}_pool`,
/// for example, `regex_pool`, `cache_pool`, etc.
pub struct PoolToy<T>
where
    T: Clone + Send,
{
    pool: Vec<T>,
}

impl<T> AsRef<T> for PoolToy<T>
where
    T: Clone + Send,
{
    fn as_ref(&self) -> &T {
        self.get()
    }
}

impl<T> PoolToy<T>
where
    T: Clone + Send,
{
    /// Create a new thread-local pool with the given vector of items.
    pub fn new(pool: Vec<T>) -> Self {
        assert!(!pool.is_empty());
        Self { pool }
    }

    /// Initialize a new thread-local pool with the given item and maximum pool size.
    pub fn init(
        item: T,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let max_pool = threads::resolve_max_pool(max_pool);

        Self::new(vec![item; max_pool])
    }

    /// Get a reference to the item for the current thread.
    pub fn get(&self) -> &T {
        let tid = threads::unstable_current_thread_id_hash();
        &self.pool[tid % self.pool.len()]
    }

    /// Get the length of the pool.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.pool.len()
    }
}

impl<T> Clone for PoolToy<T>
where
    T: Clone + Send,
{
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

impl<T> Debug for PoolToy<T>
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
        let pool = PoolToy::init(10, max_pool);

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
