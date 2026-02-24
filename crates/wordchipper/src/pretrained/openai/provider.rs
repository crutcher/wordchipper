use core::str::FromStr;
use std::sync::Arc;

use crate::{
    UnifiedTokenVocab,
    WCError,
    WCResult,
    prelude::*,
    pretrained::{VocabDescription, VocabProvider, openai::OATokenizer},
    support::resources::ResourceLoader,
};

/// [`VocabProvider`] for `OpenAI` models.
pub struct OpenaiVocabProvider {}

impl VocabProvider for OpenaiVocabProvider {
    fn id(&self) -> String {
        "OpenAI".to_string()
    }

    fn description(&self) -> String {
        "Pretrained vocabularies from OpenAI".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        vec![
            VocabDescription {
                id: "p50k_base".to_string(),
                context: vec!["openai".to_string(), "p50k_base".to_string()],
                description: "GPT-2 `p50k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "r50k_base".to_string(),
                context: vec!["openai".to_string(), "r50k_base".to_string()],
                description: "GPT-2 `r50k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "r50k_edit".to_string(),
                context: vec!["openai".to_string(), "r50k_edit".to_string()],
                description: "GPT-2 `r50k_edit` vocabulary".to_string(),
            },
            VocabDescription {
                id: "cl100k_base".to_string(),
                context: vec!["openai".to_string(), "cl100k_base".to_string()],
                description: "GPT-3 `cl100k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "o200k_base".to_string(),
                context: vec!["openai".to_string(), "o200k_base".to_string()],
                description: "GPT-5 `o200k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "o200k_harmony".to_string(),
                context: vec!["openai".to_string(), "o200k_harmony".to_string()],
                description: "GPT-5 `o200k_harmony` vocabulary".to_string(),
            },
        ]
    }

    fn load_vocab(
        &self,
        name: &str,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)> {
        let descr = self.resolve_vocab(name)?;

        if let Ok(oat) = OATokenizer::from_str(name) {
            let vocab = oat.load_vocab(loader)?;
            return Ok((descr, vocab.into()));
        }

        Err(WCError::ResourceNotFound(name.to_string()))
    }
}
