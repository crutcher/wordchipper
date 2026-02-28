//! Accelerated custom [`SpanLexer`] machinery.
use crate::{
    alloc::sync::Arc,
    spanners::span_lexers::SpanLexer,
    support::regex::ConstRegexPattern,
};

/// The [`inventory`] hook mechanism for registering regex accelerators.
///
/// These accelerators provide compiled lexers as replacements for
/// specific targeted regex patterns.
///
/// See:
/// * [`get_regex_accelerator`]
/// * [`build_regex_lexer`](`super::build_regex_lexer`)
///
/// # Example
///
/// ```rust,ignore
/// 
/// const MY_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy("abc");
///
/// struct MyRegexAccelerator {}
/// impl SpanLexer for MyRegexAccelerator { ... }
///
/// inventory::submit! {
///     RegexAcceleratorHook::new(
///         MY_PATTERN,
///         || Arc::new(MyRegexAccelerator)
///     )
/// }
/// ```
pub struct RegexAcceleratorHook {
    /// The exact regex pattern.
    pub pattern: ConstRegexPattern,

    /// The [`SpanLexer]` builder function.
    pub builder: fn() -> Arc<dyn SpanLexer>,
}
inventory::collect!(RegexAcceleratorHook);

impl RegexAcceleratorHook {
    /// Setup a new regex accelerator hook.
    pub const fn new(
        pattern: ConstRegexPattern,
        builder: fn() -> Arc<dyn SpanLexer>,
    ) -> Self {
        Self { pattern, builder }
    }
}

/// Get a regex accelerator.
///
/// ## Returns
/// - `Some(Arc<dyn SpanLexer>)` if an accelerator is found,
/// - `None` otherwise.
pub fn get_regex_accelerator(pattern: &str) -> Option<Arc<dyn SpanLexer>> {
    for hook in inventory::iter::<RegexAcceleratorHook> {
        if hook.pattern.as_str() == pattern {
            return Some((hook.builder)());
        }
    }
    None
}
