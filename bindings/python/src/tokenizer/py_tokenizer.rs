use std::sync::Arc;

use pyo3::{
    Bound,
    PyRef,
    PyResult,
    Python,
    pyclass,
    pymethods,
};
use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    VocabIndex,
};

use super::TokenizerOptions;
use crate::{
    support::to_pyerr,
    vocab::{
        _Vocab,
        SpecialFilter,
    },
    wc,
};

#[pyclass]
pub struct _Tokenizer {
    inner: Arc<wc::Tokenizer<u32>>,
}

#[pymethods]
impl _Tokenizer {
    #[staticmethod]
    #[pyo3(signature = (name, options=None))]
    fn from_pretrained(
        py: Python<'_>,
        name: &str,
        options: Option<Bound<'_, TokenizerOptions>>,
    ) -> PyResult<Self> {
        let binding: Option<PyRef<TokenizerOptions>> = options.map(|o| o.borrow());
        let options = binding.map(|o| o.clone()).unwrap_or_default();

        py.detach(|| {
            let mut disk_cache = wc::WordchipperDiskCache::default();

            let loaded = wc::load_vocab(name, &mut disk_cache).map_err(to_pyerr)?;

            let inner = options.inner().build(loaded.vocab().clone());

            Ok(_Tokenizer { inner })
        })
    }

    #[pyo3(signature = (text, special_filter=None))]
    fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        special_filter: Option<Bound<'_, SpecialFilter>>,
    ) -> PyResult<Vec<u32>> {
        let binding: Option<PyRef<SpecialFilter>> = special_filter.map(|filter| filter.borrow());
        let filter = binding.as_ref().map(|filter| filter.inner());

        py.detach(|| self.inner.try_encode(text, filter))
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (texts, special_filter=None))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        special_filter: Option<Bound<'_, SpecialFilter>>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let binding: Option<PyRef<SpecialFilter>> = special_filter.map(|filter| filter.borrow());
        let filter = binding.as_ref().map(|filter| filter.inner());

        py.detach(|| {
            let refs = wc::inner_str_view(&texts);
            self.inner.try_encode_batch(&refs, filter)
        })
        .map_err(to_pyerr)
    }

    fn decode(
        &self,
        py: Python<'_>,
        tokens: Vec<u32>,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .try_decode_to_string(&tokens)
                .and_then(|r| r.try_result())
        })
        .map_err(to_pyerr)
    }

    fn decode_batch(
        &self,
        py: Python<'_>,
        batch: Vec<Vec<u32>>,
    ) -> PyResult<Vec<String>> {
        py.detach(|| {
            let refs = wc::inner_slice_view(&batch);
            self.inner
                .try_decode_batch_to_strings(&refs)
                .and_then(|r| r.try_results())
        })
        .map_err(to_pyerr)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }

    #[getter]
    fn max_token(&self) -> Option<u32> {
        self.inner.vocab().max_token()
    }

    #[getter]
    fn vocab(&self) -> _Vocab {
        _Vocab::new(self.inner.vocab().clone())
    }

    fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.inner.vocab().lookup_token(token.as_bytes())
    }

    fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.inner
            .vocab()
            .unified_dictionary()
            .get(&id)
            .map(|bytes| wc::string_from_utf8_lossy(bytes.clone()))
    }

    fn get_special_tokens(&self) -> Vec<(String, u32)> {
        self.inner
            .vocab()
            .special_vocab()
            .span_map()
            .iter()
            .map(|(bytes, &token)| (wc::string_from_utf8_lossy(bytes.to_vec()), token))
            .collect()
    }

    #[staticmethod]
    fn available_models() -> Vec<String> {
        wordchipper::list_models()
    }

    fn save_base64_vocab(
        &self,
        py: Python<'_>,
        path: &str,
    ) -> PyResult<()> {
        py.detach(|| {
            wc::save_base64_span_map_path(self.inner.vocab().span_vocab().span_map(), path)
                .map_err(to_pyerr)
        })
    }
}
