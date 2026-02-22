//! # Span Encoder Selector

use crate::encoders::span_encoders::{
    IncrementalSweepSpanEncoder, MergeHeapSpanEncoder, SpanEncoder, TokenSpanEncoder,
};
use crate::{TokenType, UnifiedTokenVocab};
use std::sync::Arc;

#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    strum_macros::EnumString,
    strum_macros::EnumIter,
    strum_macros::Display,
)]
#[non_exhaustive]
pub enum SpanEncoderSelector {
    /// Reference encoder.
    ///
    /// See: [`IncrementalSweepSpanEncoder`]
    Reference,

    /// [`IncrementalSweepSpanEncoder`] encoder.
    IncrementalSweep,

    /// [`MergeHeapSpanEncoder`] encoder.
    #[default]
    MergeHeap,
}

impl SpanEncoderSelector {
    pub fn span_encoder_builder<T: TokenType>(
        &self
    ) -> Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync> {
        match self {
            Self::Reference | Self::IncrementalSweep => {
                Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<T>::default()))
            }
            Self::MergeHeap => Arc::new(|| Box::new(MergeHeapSpanEncoder::<T>::default())),
        }
    }

    /// Build the configured [`SpanEncoder`].
    pub fn build<T: TokenType>(
        self,
        vocab: Arc<UnifiedTokenVocab<T>>,
    ) -> Arc<TokenSpanEncoder<T>> {
        Arc::new(TokenSpanEncoder::<T>::new(
            vocab.spanning().build(),
            vocab,
            self.span_encoder_builder(),
        ))
    }
}
