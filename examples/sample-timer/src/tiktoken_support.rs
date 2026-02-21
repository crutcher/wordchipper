use std::sync::Arc;

use tiktoken_rs::{CoreBPE, Rank};

use crate::{
    ModelSelector,
    engines::{BoxError, EncDecEngine},
};

/// [`EncDecEngine`] implementation for [`CoreBPE`].
pub struct TiktokenRsEngine {
    name: String,
    inner: Arc<CoreBPE>,
}

impl TiktokenRsEngine {
    pub fn new(
        name: String,
        inner: Arc<CoreBPE>,
    ) -> Self {
        let name = format!("tiktoken-rs::{name}");
        Self { name, inner }
    }
}

impl EncDecEngine<Rank> for TiktokenRsEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> Result<Vec<Vec<Rank>>, BoxError> {
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
    ) -> Result<Vec<String>, BoxError> {
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

/// Load a tiktoken model from the given `OATokenizer` enum variant.
pub fn load_tiktoken_bpe(model: ModelSelector) -> Result<(String, Arc<CoreBPE>), BoxError> {
    use ModelSelector::*;

    let (source, bpe) = match model {
        OpenaiR50kBase => ("r50k_base", tiktoken_rs::r50k_base()?),
        OpenaiP50kBase => ("p50k_base", tiktoken_rs::p50k_base()?),
        OpenaiP50kEdit => ("p50k_edit", tiktoken_rs::p50k_edit()?),
        OpenaiCl100kBase => ("cl100k_base", tiktoken_rs::cl100k_base()?),
        OpenaiO200kBase => ("o200k_base", tiktoken_rs::o200k_base()?),
        OpenaiO200kHarmony => ("o200k_harmony", tiktoken_rs::o200k_harmony()?),
        _ => return Err(format!("unsupported model: {:?}", model).into()),
    };
    Ok((source.to_string(), Arc::new(bpe)))
}
