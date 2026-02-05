//! # Rayon Utilities
//!
//! [`rayon`] powered wrappers for parallel encoders and decoders.

mod rayon_decoder;
mod rayon_encoder;

#[doc(inline)]
pub use rayon_decoder::ParallelRayonDecoder;
#[doc(inline)]
pub use rayon_encoder::ParallelRayonEncoder;
