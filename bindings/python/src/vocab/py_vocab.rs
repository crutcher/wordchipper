use std::sync::Arc;

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
    vocab::UnifiedTokenVocab,
};

use crate::wc;

#[pyclass(name = "_Vocab")]
pub struct _Vocab {
    inner: Arc<UnifiedTokenVocab<u32>>,
}

impl _Vocab {
    pub fn new(inner: Arc<UnifiedTokenVocab<u32>>) -> Self {
        Self { inner }
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
        self.inner.len() + self.inner.special_vocab().len()
    }

    #[getter]
    fn max_token(&self) -> Option<u32> {
        let core_max = self.inner.max_token();
        let special_max = self.inner.special_vocab().max_token();
        [core_max, special_max].into_iter().flatten().max()
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
        self.inner
            .unified_dictionary()
            .get(&id)
            .map(|bytes| wc::string_from_utf8_lossy(bytes.clone()))
    }

    fn ids_to_tokens(
        &self,
        ids: Vec<u32>,
    ) -> Vec<Option<String>> {
        let dict = self.inner.unified_dictionary();
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
        for (id, bytes) in self.inner.unified_dictionary() {
            let key = wc::string_from_utf8_lossy(bytes);
            dict.set_item(key, id)?;
        }
        Ok(dict)
    }
}
