use std::sync::Arc;

use wordchipper::{TokenDecoder, TokenEncoder, TokenType, spanning::TextSpanner};

use crate::engines::EncDecEngine;

/// [`EncDecEngine`] implementation for [`TokenEncoder`] + [`TokenDecoder`].
pub struct WordchipperEngine<T: TokenType> {
    name: String,
    encoder: Arc<dyn TokenEncoder<T>>,
    decoder: Arc<dyn TokenDecoder<T>>,
}

impl<T: TokenType> WordchipperEngine<T> {
    pub fn new(
        name: String,
        encoder: Arc<dyn TokenEncoder<T>>,
        decoder: Arc<dyn TokenDecoder<T>>,
    ) -> Self {
        let name = format!("wordchipper::{name}");
        Self {
            name,
            encoder,
            decoder,
        }
    }

    pub fn spanner(&self) -> Arc<dyn TextSpanner> {
        self.encoder.spanner()
    }
}

impl<T: TokenType> EncDecEngine<T> for WordchipperEngine<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>> {
        self.encoder.try_encode_batch(batch)
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<Vec<String>> {
        Ok(self.decoder.try_decode_batch_to_strings(batch)?.unwrap())
    }
}
