//! # Vocabulary Factories

use once_cell::sync::OnceCell;
use spin::RwLock;

use crate::{
    UnifiedTokenVocab,
    WCError,
    WCResult,
    alloc::{format, string::String, sync::Arc},
    prelude::*,
    pretrained::openai,
    support::resources::ResourceLoader,
};

/// Global vocabulary factory.
static FACTORY: OnceCell<RwLock<VocabFactory>> = OnceCell::new();

fn init_factory() -> VocabFactory {
    let mut factory = VocabFactory::default();

    factory
        .register_provider(Arc::new(openai::OpenaiVocabProvider {}))
        .unwrap();

    factory
}

/// Get the global vocabulary factory.
pub fn get_vocab_factory() -> &'static RwLock<VocabFactory> {
    FACTORY.get_or_init(|| RwLock::new(init_factory()))
}

/// List all known vocabularies across all loaders.
pub fn list_vocabs() -> Vec<VocabListing> {
    let guard = get_vocab_factory().read();
    guard.list_vocabs()
}

/// Resolve a [`VocabListing`] by name.
pub fn resolve_vocab(name: &str) -> WCResult<VocabDescription> {
    let guard = get_vocab_factory().read();
    guard.resolve_vocab(name)
}

/// Load a [`UnifiedTokenVocab`] by name.
///
/// ## Returns
/// * `Ok((desc, vocab))` - on success.
/// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
/// * `Err(e)` - on any other error.
pub fn load_vocab(
    name: &str,
    loader: &mut dyn ResourceLoader,
) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)> {
    let guard = get_vocab_factory().read();
    guard.load_vocab(name, loader)
}

/// List the available pretrained models.
///
/// ## Arguments
/// * `aliases` - Whether to include all aliases or just the primary names.
pub fn list_models() -> Vec<String> {
    let mut res = Vec::new();
    for listing in list_vocabs() {
        let source = listing.source;
        for descr in &listing.vocabs {
            let name = format!("{source}::{}", descr.id.clone());
            res.push(name);
        }
    }
    res
}

/// A description of a pretrained vocabulary.
pub struct VocabDescription {
    /// The resolution id of the vocabulary.
    pub id: String,

    /// The cache context for the vocabulary.
    pub context: Vec<String>,

    /// A description of the vocabulary.
    pub description: String,
}

/// A listing of known vocabularies.
pub struct VocabListing {
    /// The id of the factory that produced the vocabularies.
    pub source: String,

    /// A description of the factory.
    pub description: String,

    /// Explicitly listed vocabularies.
    pub vocabs: Vec<VocabDescription>,
}

/// A factory for searching for and loading.
pub trait VocabProvider: Sync + Send {
    /// The name of the factory.
    fn id(&self) -> String;

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
        name: &str,
    ) -> WCResult<VocabDescription> {
        for vocab in self.list_vocabs() {
            if vocab.id == name {
                return Ok(vocab);
            }
        }
        Err(WCError::ResourceNotFound(name.to_string()))
    }

    /// Load a vocabulary from a name.
    ///
    /// ## Returns
    /// * `Ok((desc, vocab))` - on success.
    /// * `Err(WCError::ResourceNotFound)` - the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    fn load_vocab(
        &self,
        name: &str,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)>;
}

/// A factory for searching for and loading vocabularies.
#[derive(Default)]
pub struct VocabFactory {
    providers: Vec<Arc<dyn VocabProvider>>,
}

impl VocabFactory {
    /// Get a reference to the registered vocabulary providers.
    pub fn providers(&self) -> &[Arc<dyn VocabProvider>] {
        &self.providers
    }

    /// Find a provider by its id.
    pub fn find_provider(
        &self,
        id: &str,
    ) -> Option<&Arc<dyn VocabProvider>> {
        self.providers
            .iter()
            .find(|p| p.id().to_lowercase() == id.to_lowercase())
    }

    /// Register a new [`VocabProvider`].
    ///
    /// ## Returns
    /// * `Ok(())` - on success,
    /// * `Err(WCError::DuplicatedResource)` - if a provider with the same name already exists
    pub fn register_provider(
        &mut self,
        provider: Arc<dyn VocabProvider>,
    ) -> WCResult<()> {
        let id = provider.id().to_lowercase();

        for existing in &self.providers {
            if id == existing.id().to_lowercase() {
                return Err(WCError::DuplicatedResource(format!(
                    "Vocabulary provider with id '{id}' already exists",
                )));
            }
        }
        self.providers.push(provider);
        Ok(())
    }

    /// Remove a [`VocabProvider`].
    ///
    /// ## Returns
    /// The removed resource, if any.
    pub fn remove_provider(
        &mut self,
        id: &str,
    ) -> Option<Arc<dyn VocabProvider>> {
        self.providers
            .iter()
            .position(|p| p.id() == id)
            .map(|i| self.providers.remove(i))
    }

    /// List all known vocabularies across all loaders.
    pub fn list_vocabs(&self) -> Vec<VocabListing> {
        let mut res = Vec::new();
        for provider in &self.providers {
            res.push(VocabListing {
                source: provider.id(),
                description: provider.description(),
                vocabs: provider.list_vocabs(),
            });
        }
        res
    }

    /// Resolve a [`VocabDescription`] by name.
    ///
    /// ## Returns
    /// * `Ok(description)` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    pub fn resolve_vocab(
        &self,
        name: &str,
    ) -> WCResult<VocabDescription> {
        if name.contains("::") {
            let (provider_name, vocab_name) = name.split_once("::").unwrap();

            if let Some(provider) = self.find_provider(provider_name) {
                match provider.resolve_vocab(vocab_name) {
                    Ok(vocab) => return Ok(vocab),
                    Err(WCError::ResourceNotFound(_)) => {
                        return Err(WCError::ResourceNotFound(name.to_string()));
                    }
                    Err(err) => return Err(err),
                }
            }
            return Err(WCError::ResourceNotFound(name.to_string()));
        }

        for provider in &self.providers {
            match provider.resolve_vocab(name) {
                Ok(vocab) => return Ok(vocab),
                Err(WCError::ResourceNotFound(_)) => (),
                Err(err) => return Err(err),
            }
        }
        Err(WCError::ResourceNotFound(name.to_string()))
    }

    /// Load a [`UnifiedTokenVocab`] by name.
    ///
    /// ## Returns
    /// * `Ok((desc, vocab))` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    pub fn load_vocab(
        &self,
        name: &str,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)> {
        if name.contains("::") {
            let (provider_name, vocab_name) = name.split_once("::").unwrap();

            if let Some(provider) = self.find_provider(provider_name) {
                match provider.load_vocab(vocab_name, loader) {
                    Ok(vocab) => return Ok(vocab),
                    Err(WCError::ResourceNotFound(_)) => {
                        return Err(WCError::ResourceNotFound(name.to_string()));
                    }
                    Err(err) => return Err(err),
                }
            }
            return Err(WCError::ResourceNotFound(name.to_string()));
        }

        for provider in &self.providers {
            match provider.load_vocab(name, loader) {
                Ok(vocab) => return Ok(vocab),
                Err(WCError::ResourceNotFound(_)) => (),
                Err(err) => return Err(err),
            }
        }
        Err(WCError::ResourceNotFound(name.to_string()))
    }
}
