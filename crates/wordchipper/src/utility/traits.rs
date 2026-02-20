//! # Trait Compatibility and Utility

/// Static if a type is `Send`.
pub fn static_is_send_check<S: Send>(_: &S) {}

/// Check if a type is `Sync`.
pub fn static_is_sync_check<S: Sync>(_: &S) {}

/// Check if a type is `Send` and `Sync`.
pub fn static_is_send_sync_check<S: Send + Sync>(v: &S) {
    static_is_send_check(v);
    static_is_sync_check(v);
}
