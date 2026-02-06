//! # wordchipper Regex Utilities
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
#![warn(missing_docs, unused)]

extern crate alloc;

mod alternate_choice;
mod regex_wrapper;

#[doc(inline)]
pub use crate::alternate_choice::*;
#[doc(inline)]
pub use crate::regex_wrapper::*;
