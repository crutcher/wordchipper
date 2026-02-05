//! # Trainer Implementation Utilities

mod pair_span_index;
#[doc(inline)]
pub use pair_span_index::{PairCountMap, PairIndexMap, PairSpanIndex};

mod text_span_counter;
#[doc(inline)]
pub use text_span_counter::{TextSpanCounter, TextSpanCounterOptions};

mod token_span_buffer;
#[doc(inline)]
pub use token_span_buffer::TokenSpanBuf;
