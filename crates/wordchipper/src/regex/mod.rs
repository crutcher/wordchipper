//! # Regex Utilities
//!
//! This module attempts to balance two problems:
//! * Pattern Complexity
//! * Concurrence Contention
//!
//! ### Pattern Complexity
//!
//! A number of popular in-use LLM Tokenizer Regex Patterns require extended regex
//! machinery provided by the [`fancy_regex`] crate; but naturally, this has performance
//! costs. We'd prefer to avoid using the [`fancy_regex`] crate when possible, falling back
//! on the standard [`regex`] crate when patterns permit this.
//!
//! This recurses into two problems:
//!
//! * Labeling Patterns - [`RegexWrapperPattern`]
//!   * [`RegexWrapperPattern::Basic`] - a pattern which was written for [`regex`].
//!   * [`RegexWrapperPattern::Fancy`] - a pattern which was written for [`fancy_regex`].
//!   * [`RegexWrapperPattern::Adaptive`] - unknown target, try basic; then fall-up to fancy.
//! * Wrapping Compiled Regex - [`RegexWrapper`]
//!
//! The [`RegexWrapper`] type supports only one operation, ``find_iter()`` which requires
//! some adaptation of the `Iterator` stream to function.
//!
//! ### Concurrence Contention
//!
//! There are some observed thread contentions deep in compiled regex objects,
//! short fights over shared internal buffers. In high parallelism, heavy-regex workloads,
//! this can have a large performance impact.
//!
//! At the same time, per-thread local data structures, locks, and cloning introduce
//! dependencies which may not be appropriate in all environments.
//!
//! The chosen solution to this is the combination of:
//! * [`RegexSupplier`] / [`RegexSupplierHandle`]
//! * [`regex_pool_supplier`].
//!
//! Users of a [`RegexWrapper`] that *may* be under heavy thread contention should use
//! [`regex_pool_supplier`]; which in some build environments will provide
//! a thread local clone regex supplier, and in some, a simple clone implementation.

pub mod exact_match_union;
#[cfg(feature = "std")]
pub mod regex_pool;

pub mod regex_supplier;
pub mod regex_wrapper;

pub use regex_supplier::{RegexSupplier, RegexSupplierHandle};
pub use regex_wrapper::{ErrorWrapper, RegexWrapper, RegexWrapperHandle, RegexWrapperPattern};

/// Build a [`RegexSupplierHandle`].
///
/// ## Arguments
/// * `regex` - The regex wrapper handle.
///
/// ## Returns
/// A `RegexSupplierHandle`.
pub fn default_regex_supplier(regex: RegexWrapperHandle) -> RegexSupplierHandle {
    regex
}

/// Build a regex supplier for (potentially) parallel execution.
///
/// Users of a [`RegexWrapper`] that *may* be under heavy thread contention should use
/// [`regex_pool_supplier`]; which in some build environments will provide
/// a thread local clone regex supplier, and in some, a simple clone implementation.
///
/// ## Arguments
/// * `regex` - The regex wrapper handle.
///
/// ## Returns
/// A `RegexSupplierHandle`.
pub fn regex_pool_supplier(regex: RegexWrapperHandle) -> RegexSupplierHandle {
    #[cfg(feature = "std")]
    return alloc::sync::Arc::new(regex_pool::RegexWrapperPool::new(regex));

    #[cfg(not(feature = "std"))]
    regex
}
