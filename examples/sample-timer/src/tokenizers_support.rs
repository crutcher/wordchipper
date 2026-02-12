use std::sync::Arc;

use anyhow::bail;
use tokenizers::{Encoding, tokenizer::Tokenizer};
use wordchipper::pretrained::openai::OATokenizer;

use crate::engines::EncDecEngine;

/// [`EncDecEngine`] implementation for [`Tokenizer`].
pub struct TokenizersEngine {
    name: String,
    inner: Arc<Tokenizer>,
}

impl TokenizersEngine {
    pub fn new(
        name: String,
        inner: Arc<Tokenizer>,
    ) -> Self {
        let name = format!("tokenizers::{name}");
        Self { name, inner }
    }
}

impl EncDecEngine<u32> for TokenizersEngine {
    fn name(&self) -> &str {
        &self.name
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

pub fn load_tokenizers_tok(model: OATokenizer) -> anyhow::Result<(String, Arc<Tokenizer>)> {
    use OATokenizer::*;
    let source = match model {
        R50kBase => "Xenova/gpt-3",
        P50kBase | P50kEdit => "Xenova/text-davinci-002",
        Cl100kBase => "Xenova/text-embedding-ada-002",
        O200kBase | O200kHarmony => "Xenova/gpt-4o",
        _ => bail!("unsupported model: {:?}", model),
    };

    let tok = Tokenizer::from_pretrained(source, None)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {}: {}", source, e))?;

    Ok((source.to_string(), Arc::new(tok)))
}
