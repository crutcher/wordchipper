//! # Span Encoders

mod compound_span_encoder;
mod merge_heap_encoder;
mod merge_scan_encoder;
mod span_policy;

#[doc(inline)]
pub use compound_span_encoder::CompoundSpanVocabEncoder;
#[doc(inline)]
pub use merge_heap_encoder::{MergeHeapSpanPolicy, MergeHeapVocabEncoder};
#[doc(inline)]
pub use merge_scan_encoder::{MergeScanCompoundPolicy, MergeScanVocabEncoder};
#[doc(inline)]
pub use span_policy::SpanPolicy;
