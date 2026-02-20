use std::sync::Arc;

use wordchipper::{TokenDecoder, TokenEncoder, TokenType, spanners::TextSpanner};

use crate::engines::{BoxError, EncDecEngine};

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
    ) -> Result<Vec<Vec<T>>, BoxError> {
        Ok(self.encoder.try_encode_batch(batch)?)
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> Result<Vec<String>, BoxError> {
        let decoded = self.decoder.try_decode_batch_to_strings(batch)?;
        Ok(decoded.unwrap())
    }
}
