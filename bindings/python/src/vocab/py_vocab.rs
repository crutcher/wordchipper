use std::sync::{Arc, OnceLock};

use pyo3::{
    Bound,
    PyResult,
    Python,
    exceptions::PyKeyError,
    pyclass,
    pymethods,
    types::{
        PyDict,
        PyDictMethods,
    },
};
use wordchipper::{
    VocabIndex,
    vocab::{TokenSpanMap, UnifiedTokenVocab},
};

use crate::wc;

/// Cached values computed lazily from the immutable vocabulary.
struct VocabCache {
    n_vocab: usize,
    max_token: Option<u32>,
    dictionary: TokenSpanMap<u32>,
}

#[pyclass(name = "_Vocab")]
pub struct _Vocab {
    inner: Arc<UnifiedTokenVocab<u32>>,
    cache: OnceLock<VocabCache>,
}

impl _Vocab {
    pub fn new(inner: Arc<UnifiedTokenVocab<u32>>) -> Self {
        Self {
            inner,
            cache: OnceLock::new(),
        }
    }

    fn cache(&self) -> &VocabCache {
        self.cache.get_or_init(|| {
            let n_vocab = self.inner.len() + self.inner.special_vocab().len();
            let core_max = self.inner.max_token();
            let special_max = self.inner.special_vocab().max_token();
            let max_token = [core_max, special_max].into_iter().flatten().max();
            let dictionary = self.inner.unified_dictionary();
            VocabCache {
                n_vocab,
                max_token,
                dictionary,
            }
        })
    }
}

#[pymethods]
impl _Vocab {
    fn __len__(&self) -> usize {
        self.n_vocab()
    }

    fn __contains__(
        &self,
        token: &str,
    ) -> bool {
        self.inner.lookup_token(token.as_bytes()).is_some()
            || self
                .inner
                .special_vocab()
                .lookup_token(token.as_bytes())
                .is_some()
    }

    fn __getitem__(
        &self,
        token: &str,
    ) -> PyResult<u32> {
        if let Some(id) = self.inner.lookup_token(token.as_bytes()) {
            return Ok(id);
        }
        if let Some(id) = self.inner.special_vocab().lookup_token(token.as_bytes()) {
            return Ok(id);
        }
        Err(PyKeyError::new_err(token.to_string()))
    }

    #[getter]
    fn n_vocab(&self) -> usize {
        self.cache().n_vocab
    }

    #[getter]
    fn max_token(&self) -> Option<u32> {
        self.cache().max_token
    }

    fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.inner
            .lookup_token(token.as_bytes())
            .or_else(|| self.inner.special_vocab().lookup_token(token.as_bytes()))
    }

    fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.cache()
            .dictionary
            .get(&id)
            .map(|bytes| wc::string_from_utf8_lossy(bytes.clone()))
    }

    fn ids_to_tokens(
        &self,
        ids: Vec<u32>,
    ) -> Vec<Option<String>> {
        let dict = &self.cache().dictionary;
        ids.iter()
            .map(|id| {
                dict.get(id)
                    .map(|bytes| wc::string_from_utf8_lossy(bytes.clone()))
            })
            .collect()
    }

    fn get_special_tokens(&self) -> Vec<(String, u32)> {
        self.inner
            .special_vocab()
            .span_map()
            .iter()
            .map(|(bytes, &token)| (wc::string_from_utf8_lossy(bytes.to_vec()), token))
            .collect()
    }

    fn to_dict<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (id, bytes) in &self.cache().dictionary {
            let key = wc::string_from_utf8_lossy(bytes.clone());
            dict.set_item(key, id)?;
        }
        Ok(dict)
    }
}
