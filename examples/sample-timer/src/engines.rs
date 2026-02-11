//! Timing Candidate Wrappers

use std::sync::Arc;

use tiktoken_rs::{CoreBPE, Rank};
use wordchipper::{
    decoders::TokenDecoder,
    encoders::TokenEncoder,
    spanning::TextSpanner,
    types::TokenType,
};

pub trait EncDecEngine<T: TokenType> {
    fn name(&self) -> &str;

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>>;

    fn expect_encode_batch(
        &self,
        batch: &[&str],
    ) -> Vec<Vec<T>> {
        self.encode_batch(batch)
            .unwrap_or_else(|_| panic!("failed to encode batch with \"{}\"", self.name()))
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<Vec<String>>;

    fn expect_decode_batch(
        &self,
        batch: &[&[T]],
    ) -> Vec<String> {
        self.decode_batch(batch)
            .unwrap_or_else(|_| panic!("failed to decode batch with \"{}\"", self.name()))
    }
}

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
        Self {
            name,
            encoder,
            decoder,
        }
    }

    pub fn spanner(&self) -> &TextSpanner {
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
pub struct TiktokenRsEngine {
    inner: Arc<CoreBPE>,
}

impl TiktokenRsEngine {
    pub fn new(inner: Arc<CoreBPE>) -> Self {
        Self { inner }
    }
}

impl EncDecEngine<Rank> for TiktokenRsEngine {
    fn name(&self) -> &str {
        "tiktoken-rs"
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<Rank>>> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "rayon")] {
                use rayon::prelude::*;
                let it =batch.par_iter();
            } else {
                let it =batch.iter();
            }
        }
        Ok(it
            .map(|s| self.inner.encode_with_special_tokens(s))
            .collect::<Vec<_>>())
    }

    fn decode_batch(
        &self,
        batch: &[&[Rank]],
    ) -> anyhow::Result<Vec<String>> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "rayon")] {
                use rayon::prelude::*;
                let it =batch.par_iter();
            } else {
                let it =batch.iter();
            }
        }
        Ok(it
            .map(|tokens| self.inner.decode(tokens.to_vec()).unwrap())
            .collect::<Vec<_>>())
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "tokenizers")] {
use tokenizers::Encoding;

pub struct TokenizersEngine {
    inner: Arc<tokenizers::tokenizer::Tokenizer>,
}

impl TokenizersEngine {
    pub fn new(inner: Arc<tokenizers::tokenizer::Tokenizer>) -> Self {
        Self { inner }
    }
}

impl EncDecEngine<u32> for TokenizersEngine {
    fn name(&self) -> &str {
        "tokenizers"
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let batch = batch.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let br = self.inner.encode_batch(batch, true).unwrap();
        Ok(br
            .iter()
            .map(|e: &Encoding| e.get_ids().to_vec())
            .collect::<Vec<_>>())
    }

    fn decode_batch(
        &self,
        batch: &[&[u32]],
    ) -> anyhow::Result<Vec<String>> {
        match self.inner.decode_batch(batch, false) {
            Ok(res) => Ok(res),
            Err(e) => Err(anyhow::anyhow!(
                "failed to decode batch with \"{}\": {}",
                self.name(),
                e
            )),
        }
    }
}
    }
}
