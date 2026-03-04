use crate::{
    UnifiedTokenVocab,
    WCError,
    WCResult,
    alloc::sync::Arc,
    prelude::*,
    pretrained::{
        LabeledVocab,
        VocabDescription,
        VocabQuery,
        factory::VocabProviderInventoryHook,
    },
    support::resources::ResourceLoader,
};

/// A factory for searching for and loading.
pub trait VocabProvider: Sync + Send {
    /// The name of the factory.
    fn name(&self) -> String;

    /// Get an extended description of the factory.
    fn description(&self) -> String;

    /// Get a listing of known vocabularies.
    fn list_vocabs(&self) -> Vec<VocabDescription>;

    /// Resolve a vocabulary description.
    ///
    /// ## Returns
    /// * `Ok(description)` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    fn resolve_vocab(
        &self,
        query: &VocabQuery,
    ) -> WCResult<VocabDescription> {
        for desc in self.list_vocabs() {
            if desc.id().fuzzy_match(query) {
                return Ok(desc);
            }
        }
        Err(WCError::ResourceNotFound(query.to_string()))
    }

    /// Load a vocabulary from a name.
    ///
    /// ## Returns
    /// * `Ok((desc, vocab))` - on success.
    /// * `Err(WCError::ResourceNotFound)` - the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    fn load_vocab(
        &self,
        query: &VocabQuery,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>>;
}

/// Hook for registering pretrained vocabularies.
pub struct BuiltinPretrainedVocabHook {
    id: &'static str,

    /// Function to create the vocabulary description.
    descr_fn: fn(&str) -> VocabDescription,

    /// The function to load the vocabulary.
    #[allow(clippy::type_complexity)]
    vocab_fn: fn(&VocabDescription, &mut dyn ResourceLoader) -> WCResult<UnifiedTokenVocab<u32>>,
}
inventory::collect!(BuiltinPretrainedVocabHook);

impl BuiltinPretrainedVocabHook {
    /// Build a new hook.
    #[allow(clippy::type_complexity)]
    pub const fn new(
        id: &'static str,
        descr_fn: fn(&str) -> VocabDescription,
        vocab_fn: fn(
            &VocabDescription,
            &mut dyn ResourceLoader,
        ) -> WCResult<UnifiedTokenVocab<u32>>,
    ) -> Self {
        Self {
            id,
            descr_fn,
            vocab_fn,
        }
    }

    /// The ID of the vocabulary.
    pub fn id(&self) -> &str {
        self.id
    }

    /// The description of the vocabulary.
    pub fn description(&self) -> VocabDescription {
        (self.descr_fn)(self.id)
    }

    /// The vocab loader callback.
    pub fn vocab_fn(
        &self
    ) -> &fn(&VocabDescription, &mut dyn ResourceLoader) -> WCResult<UnifiedTokenVocab<u32>> {
        &self.vocab_fn
    }
}

/// [`VocabProvider`] for [`BuiltinPretrainedVocabHook`].
pub struct BuiltinVocabProvider {}

inventory::submit! {
    VocabProviderInventoryHook::new(|| Arc::new(BuiltinVocabProvider{}))
}

impl VocabProvider for BuiltinVocabProvider {
    fn name(&self) -> String {
        "builtin".to_string()
    }

    fn description(&self) -> String {
        "Link-registered vocabularies".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        let mut res = Vec::new();
        for hook in inventory::iter::<BuiltinPretrainedVocabHook> {
            res.push(hook.description());
        }
        res
    }

    fn load_vocab(
        &self,
        query: &VocabQuery,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>> {
        for hook in inventory::iter::<BuiltinPretrainedVocabHook> {
            let description = hook.description();
            if description.id().fuzzy_match(query) {
                let vocab = (hook.vocab_fn)(&description, loader)?;
                return Ok(LabeledVocab::new(description, vocab.into()));
            }
        }
        Err(WCError::ResourceNotFound(query.to_string()))
    }
}
