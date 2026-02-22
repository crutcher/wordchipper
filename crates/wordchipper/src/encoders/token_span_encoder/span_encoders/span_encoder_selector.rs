//! # Span Encoder Selector

use crate::{
    TokenType,
    alloc::{boxed::Box, sync::Arc},
    encoders::token_span_encoder::{
        SpanEncoder,
        span_encoders::{
            BufferSweepSpanEncoder,
            MergeHeapSpanEncoder,
            PriorityMergeSpanEncoder,
            TailSweepSpanEncoder,
        },
    },
};

/// Policy enum for selecting a [`SpanEncoder`] for
/// [`TokenSpanEncoder`](`crate::encoders::token_span_encoder::TokenSpanEncoder`).
#[derive(Default, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "std",
    derive(
        strum_macros::EnumString,
        strum_macros::EnumIter,
        strum_macros::Display
    )
)]
#[non_exhaustive]
pub enum SpanEncoderSelector {
    /// The canonical best Default encoder.
    ///
    /// Users should, in general, prefer to use this encoder.
    ///
    /// As improved encoders are developed, this will remain the evergreen
    /// label for "the good one". We expose this, rather than making
    /// a particular encoder the marked default, so that serializations
    /// of this policy into config files will convey "use the default",
    /// rather than "use the default as of the time this config was saved".
    ///
    /// This is currently an alias for: [`PriorityMerge`](`Self::PriorityMerge`)
    #[default]
    Default,

    /// The canonical reference encoder, [`BufferSweepSpanEncoder`].
    ///
    /// This encoder is meant to be used as a reference implementation for testing and comparison.
    /// The code and behavior are as simple as possible, but it is not optimized for performance.
    ///
    /// This is currently an alias for: [`BufferSweep`](`Self::BufferSweep`)
    Reference,

    /// Use the [`TailSweepSpanEncoder`] encoder.
    TailSweep,

    /// Use the [`MergeHeapSpanEncoder`] encoder.
    MergeHeap,

    /// Use the [`PriorityMergeSpanEncoder`] encoder.
    PriorityMerge,

    /// Use the [`BufferSweepSpanEncoder`] encoder.
    BufferSweep,
}

impl SpanEncoderSelector {
    /// Get a builder for the configured [`SpanEncoder`].
    pub fn span_encoder_builder<T: TokenType>(
        &self
    ) -> Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync> {
        use SpanEncoderSelector::*;
        match self {
            Reference | BufferSweep => {
                Arc::new(|| Box::new(BufferSweepSpanEncoder::<T>::default()))
            }
            TailSweep => Arc::new(|| Box::new(TailSweepSpanEncoder::<T>::default())),
            MergeHeap => Arc::new(|| Box::new(MergeHeapSpanEncoder::<T>::default())),
            Default | PriorityMerge => {
                Arc::new(|| Box::new(PriorityMergeSpanEncoder::<T>::default()))
            }
        }
    }
}
