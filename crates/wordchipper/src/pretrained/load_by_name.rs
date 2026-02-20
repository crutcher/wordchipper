use crate::{
    UnifiedTokenVocab,
    alloc::{string::String, vec::Vec},
    errors::WordchipperError,
    pretrained::openai::OATokenizer,
    utility::resources::ResourceLoader,
};

/// A hook that can be used to load pretrained models.
pub struct ConstPretrainedHook {
    /// The aliases for the pretrained model.
    pub aliases: &'static [&'static str],

    /// A function that loads the pretrained model.
    pub load: fn(&str, &mut dyn ResourceLoader) -> crate::errors::Result<UnifiedTokenVocab<u32>>,
}

const PRETRAINED_HOOKS: &[ConstPretrainedHook] = &[
    ConstPretrainedHook {
        aliases: &["openai/r50k_base", "r50k_base"],
        load: |_, loader| OATokenizer::R50kBase.load_vocab(loader),
    },
    ConstPretrainedHook {
        aliases: &["openai/p50k_base", "p50k_base"],
        load: |_, loader| OATokenizer::P50kBase.load_vocab(loader),
    },
    ConstPretrainedHook {
        aliases: &["openai/p50k_edit", "p50k_edit"],
        load: |_, loader| OATokenizer::P50kEdit.load_vocab(loader),
    },
    ConstPretrainedHook {
        aliases: &["openai/cl100k_base", "cl100k_base"],
        load: |_, loader| OATokenizer::Cl100kBase.load_vocab(loader),
    },
    ConstPretrainedHook {
        aliases: &["openai/o200k_base", "o200k_base"],
        load: |_, loader| OATokenizer::O200kBase.load_vocab(loader),
    },
    ConstPretrainedHook {
        aliases: &["openai/o200k_harmony", "o200k_harmony"],
        load: |_, loader| OATokenizer::O200kHarmony.load_vocab(loader),
    },
];

/// Load a pretrained model by name.
pub fn get_model(
    name: &str,
    loader: &mut dyn ResourceLoader,
) -> crate::errors::Result<UnifiedTokenVocab<u32>> {
    for hook in PRETRAINED_HOOKS {
        if hook.aliases.contains(&name) {
            return (hook.load)(name, loader);
        }
    }

    Err(WordchipperError::External(crate::alloc::format!(
        "Unable to load pretrained model: {name}"
    )))
}

/// List the available pretrained models.
///
/// ## Arguments
/// * `aliases` - Whether to include all aliases or just the primary names.
pub fn list_models(aliases: bool) -> Vec<String> {
    let mut models = Vec::new();
    for hook in PRETRAINED_HOOKS {
        if aliases {
            models.extend(hook.aliases.iter().map(|a| a.to_string()));
        } else {
            models.push(hook.aliases[0].to_string());
        }
    }
    models
}
