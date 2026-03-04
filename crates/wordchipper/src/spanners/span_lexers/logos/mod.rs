//! # Logos-based lexers
//!
//! Composable building blocks for compile-time DFA word scanners using
//! the [`logos`](https://docs.rs/logos) crate.
//!
//! ## Building a custom lexer
//!
//! 1. Define a `#[derive(Logos)]` enum with your token patterns.
//! 2. Map each variant to a [`Gpt2FamilyTokenRole`] (via a `role()` method or
//!    similar).
//! 3. Implement [`SpanLexer`](super::SpanLexer) by feeding the token stream to
//!    [`for_each_classified_span`].
//!
//! See [`Cl100kLexer`] and [`O200kLexer`] for reference implementations.

/// Define a [`SpanLexer`](super::SpanLexer) backed by a logos token enum.
///
/// Generates a struct, its `SpanLexer` impl (using [`logos_span_iter`]),
/// and an [`inventory::submit!`] registration for the regex accelerator.
macro_rules! logos_lexer {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident;
        token = $token:ty;
        pattern = $pattern:expr;
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        $vis struct $name;

        inventory::submit! {
            crate::spanners::span_lexers::accelerators::RegexAcceleratorHook::new(
                $pattern,
                || alloc::sync::Arc::new($name),
            )
        }

        impl crate::spanners::span_lexers::SpanLexer for $name {
            fn find_span_iter<'a>(
                &'a self,
                text: &'a str,
            ) -> alloc::boxed::Box<
                dyn Iterator<Item = core::ops::Range<usize>> + 'a,
            > {
                alloc::boxed::Box::new(
                    super::gpt2_family::logos_span_iter(
                        text,
                        <$token as logos::Logos>::lexer(text).spanned(),
                    ),
                )
            }
        }
    };
}

pub mod cl100k;
pub mod gpt2_family;
pub mod o200k;
pub mod r50k;
