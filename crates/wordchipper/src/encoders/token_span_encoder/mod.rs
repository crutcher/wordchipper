//! # [`TokenSpanEncoder`] Utilities

pub mod span_encoders;

#[allow(clippy::module_inception)]
mod token_span_encoder;

#[doc(inline)]
pub use span_encoders::{SpanEncoder, SpanEncoderSelector};
#[doc(inline)]
pub use token_span_encoder::*;
