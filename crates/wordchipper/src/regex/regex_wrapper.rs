//! # Regex Wrapper
//! This modules provides mechanisms to mix `regex` and `fancy_regex` types.

use crate::alloc::string::String;
use crate::alloc::string::ToString;
use crate::alloc::sync::Arc;
use core::fmt::Debug;

/// Error wrapper for regex patterns.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ErrorWrapper {
    /// Error from `regex`.
    Basic(regex::Error),

    /// Error from `fancy_regex`.
    Fancy(fancy_regex::Error),
}

impl From<regex::Error> for ErrorWrapper {
    fn from(err: regex::Error) -> Self {
        Self::Basic(err)
    }
}

impl From<fancy_regex::Error> for ErrorWrapper {
    fn from(err: fancy_regex::Error) -> Self {
        Self::Fancy(err)
    }
}

impl core::fmt::Display for ErrorWrapper {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            Self::Basic(err) => core::fmt::Display::fmt(err, f),
            Self::Fancy(err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl core::error::Error for ErrorWrapper {}

/// Const Regex Wrapper Pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstRegexWrapperPattern {
    /// This is a pattern for the `regex` crate.
    Basic(&'static str),

    /// This is a pattern for the `fancy_regex` crate.
    Fancy(&'static str),
}

impl ConstRegexWrapperPattern {
    /// Get the underlying regex pattern.
    ///
    /// ## Returns
    /// The regex pattern string slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Basic(pattern) => pattern,
            Self::Fancy(pattern) => pattern,
        }
    }

    /// Convert to [`RegexWrapperPattern`]
    ///
    /// ## Returns
    /// A new `RegexWrapperPattern` instance.
    pub fn to_pattern(self) -> RegexWrapperPattern {
        self.into()
    }

    /// Compile the regex pattern into a `RegexWrapper`.
    ///
    /// ## Returns
    /// A `Result` containing the compiled `RegexWrapper` or an `ErrorWrapper`.
    pub fn compile(&self) -> Result<RegexWrapper, ErrorWrapper> {
        RegexWrapperPattern::from(*self).compile()
    }
}

impl From<ConstRegexWrapperPattern> for RegexWrapperPattern {
    fn from(pattern: ConstRegexWrapperPattern) -> Self {
        use ConstRegexWrapperPattern::*;
        match pattern {
            Basic(pattern) => RegexWrapperPattern::Basic(pattern.to_string()),
            Fancy(pattern) => RegexWrapperPattern::Fancy(pattern.to_string()),
        }
    }
}

/// Label for regex patterns.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RegexWrapperPattern {
    /// This is a pattern for the `regex` crate.
    Basic(String),

    /// This is a pattern for the `fancy_regex` crate.
    Fancy(String),

    /// This pattern will try the `regex` crate first,
    /// and fallback to `fancy_regex` if it fails.
    Adaptive(String),
}

impl<S: AsRef<str>> From<S> for RegexWrapperPattern {
    fn from(pattern: S) -> Self {
        Self::Adaptive(pattern.as_ref().to_string())
    }
}

impl RegexWrapperPattern {
    /// Get the underlying regex pattern.
    ///
    /// ## Returns
    /// The regex pattern string slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Basic(pattern) => pattern,
            Self::Fancy(pattern) => pattern,
            Self::Adaptive(pattern) => pattern,
        }
    }

    /// Compile the regex pattern into a `RegexWrapper`.
    ///
    /// ## Returns
    /// A `Result` containing the compiled `RegexWrapper` or an `ErrorWrapper`.
    pub fn compile(&self) -> Result<RegexWrapper, ErrorWrapper> {
        match self {
            Self::Basic(pattern) => regex::Regex::new(pattern)
                .map(RegexWrapper::from)
                .map_err(ErrorWrapper::from),
            Self::Fancy(pattern) => fancy_regex::Regex::new(pattern)
                .map(RegexWrapper::from)
                .map_err(ErrorWrapper::from),
            Self::Adaptive(pattern) => {
                regex::Regex::new(pattern)
                    .map(RegexWrapper::from)
                    .or_else(|_| {
                        fancy_regex::Regex::new(pattern)
                            .map(RegexWrapper::from)
                            .map_err(ErrorWrapper::from)
                    })
            }
        }
    }
}

/// Common Regex Wrapper Handle Type
pub type RegexWrapperHandle = Arc<RegexWrapper>;

impl From<RegexWrapperPattern> for RegexWrapperHandle {
    fn from(val: RegexWrapperPattern) -> Self {
        Arc::new(val.compile().unwrap())
    }
}

/// Wrapper for regex patterns.
#[derive(Debug, Clone)]
pub enum RegexWrapper {
    /// Wrapper for `regex::Regex`.
    Basic(regex::Regex),

    /// Wrapper for `fancy_regex::Regex`.
    Fancy(fancy_regex::Regex),
}

impl From<regex::Regex> for RegexWrapper {
    fn from(regex: regex::Regex) -> Self {
        Self::Basic(regex)
    }
}

impl From<fancy_regex::Regex> for RegexWrapper {
    fn from(regex: fancy_regex::Regex) -> Self {
        Self::Fancy(regex)
    }
}

impl RegexWrapper {
    /// Is this `Basic`?
    ///
    /// ## Returns
    /// `true` if it wraps a `regex::Regex`, `false` otherwise.
    pub fn is_basic(&self) -> bool {
        match self {
            Self::Basic(_) => true,
            Self::Fancy(_) => false,
        }
    }

    /// Is this `Fancy`?
    ///
    /// ## Returns
    /// `true` if it wraps a `fancy_regex::Regex`, `false` otherwise.
    pub fn is_fancy(&self) -> bool {
        match self {
            Self::Basic(_) => false,
            Self::Fancy(_) => true,
        }
    }

    /// Get the underlying regex pattern.
    ///
    /// ## Returns
    /// The regex pattern string slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Basic(regex) => regex.as_str(),
            Self::Fancy(regex) => regex.as_str(),
        }
    }

    /// Wrapper for `find_iter`.
    ///
    /// ## Arguments
    /// * `haystack` - The string to search in.
    ///
    /// ## Returns
    /// A `MatchesWrapper` iterator over the matches.
    pub fn find_iter<'r, 'h>(
        &'r self,
        haystack: &'h str,
    ) -> MatchesWrapper<'r, 'h> {
        match self {
            Self::Basic(regex) => regex.find_iter(haystack).into(),
            Self::Fancy(regex) => regex.find_iter(haystack).into(),
        }
    }
}

/// Wrapper for regex matches.
pub enum MatchesWrapper<'r, 'h> {
    /// Wrapper for `regex::Matches`.
    Regex(regex::Matches<'r, 'h>),

    /// Wrapper for `fancy_regex::Matches`.
    FancyRegex(fancy_regex::Matches<'r, 'h>),
}

impl<'r, 'h> From<regex::Matches<'r, 'h>> for MatchesWrapper<'r, 'h> {
    fn from(matches: regex::Matches<'r, 'h>) -> Self {
        Self::Regex(matches)
    }
}

impl<'r, 'h> From<fancy_regex::Matches<'r, 'h>> for MatchesWrapper<'r, 'h> {
    fn from(matches: fancy_regex::Matches<'r, 'h>) -> Self {
        Self::FancyRegex(matches)
    }
}

impl<'r, 'h> Iterator for MatchesWrapper<'r, 'h> {
    type Item = regex::Match<'h>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Regex(matches) => matches.next(),
            Self::FancyRegex(matches) => matches
                .next()
                .map(|m| unsafe { core::mem::transmute(m.unwrap()) }),
        }
    }
}
