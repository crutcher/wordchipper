//! Parser for divan benchmark output.
//!
//! Divan has no machine-readable output format. This module parses the
//! human-readable table output line by line and produces [`BenchResult`]
//! values.

mod divan_parser;
mod result_types;
pub use divan_parser::*;
pub use result_types::*;
