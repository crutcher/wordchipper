//! Token Decoder Options
//!
//! Options for building a [`TokenDecoder`].

use crate::{
    TokenDecoder,
    TokenType,
    UnifiedTokenVocab,
    alloc::sync::Arc,
    decoders::SlabIndexDecoder,
};

/// Options for configuring a [`TokenDecoder`].
// TODO: serialize/deserialize?
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TokenDecoderOptions {
    /// Should the decoder be threaded?
    ///
    /// Enabling parallelism will request a threaded implementation.
    pub parallel: bool,
}

impl TokenDecoderOptions {
    /// Gets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded implementation.
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded implementation.
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.parallel = parallel;
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request a threaded implementation.
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Build a [`TokenDecoder`] for the given vocab.
    pub fn build<T: TokenType>(
        &self,
        vocab: Arc<UnifiedTokenVocab<T>>,
    ) -> Arc<dyn TokenDecoder<T>> {
        #[allow(unused_mut)]
        let mut dec: Arc<dyn TokenDecoder<T>> = Arc::new(SlabIndexDecoder::from_vocab(vocab));

        #[cfg(feature = "parallel")]
        if self.parallel {
            use crate::support::concurrency::rayon::ParallelRayonDecoder;
            dec = Arc::new(ParallelRayonDecoder::new(dec));
        }

        dec
    }
}
