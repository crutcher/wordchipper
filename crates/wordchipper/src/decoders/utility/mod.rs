//! # Decoder Testing Utilities

mod byte_decoder;
mod pair_decoder;

#[doc(inline)]
pub use byte_decoder::ByteDecoder;
#[doc(inline)]
pub use pair_decoder::PairExpansionDecoder;

#[cfg(test)]
pub mod testing;
