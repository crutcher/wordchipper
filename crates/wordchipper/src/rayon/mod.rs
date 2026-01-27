//! # Rayon Utilities
//!
//! [`rayon`] powered wrappers for parallel encoders and decoders.

mod rayon_decoder;
mod rayon_encoder;

pub use rayon_decoder::ParallelRayonDecoder;
pub use rayon_encoder::ParallelRayonEncoder;
