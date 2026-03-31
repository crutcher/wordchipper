use std::{
    collections::HashMap,
    sync::Arc,
};

use pyo3::{
    Bound,
    PyRef,
    PyResult,
    Python,
    pyclass,
    pymethods,
    types::PyBytes,
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

    /// Load a tokenizer from a tiktoken-format vocabulary file.
    ///
    /// Parameters
    /// ----------
    /// path:
    ///     Path to a ``.tiktoken`` file (lines of ``BASE64_BYTES TOKEN_ID``).
    /// pattern:
    ///     Regex pattern used to pre-split text before BPE encoding.
    ///     Lookaheads and possessive quantifiers are supported via fancy-regex.
    /// special_tokens:
    ///     Optional mapping of special-token strings to their integer IDs.
    /// options:
    ///     Tokenizer options (parallelism, accelerated lexers, …).
    #[staticmethod]
    #[pyo3(signature = (path, pattern, special_tokens=None, options=TokenizerOptions::default()))]
    fn from_tiktoken_file(
        py: Python<'_>,
        path: &str,
        pattern: &str,
        special_tokens: Option<HashMap<String, u32>>,
        options: TokenizerOptions,
    ) -> PyResult<Self> {
        let pattern = pattern.to_string();
        let special_tokens = special_tokens.unwrap_or_default();
        let path = path.to_string();
        py.detach(|| {
            let regex = wc::RegexPattern::Adaptive(pattern);
            let specials: Vec<(String, u32)> = special_tokens.into_iter().collect();
            let spanning = wc::TextSpanningConfig::<u32>::from(regex)
                .with_special_words(specials);

            // Load the raw span map and strip any entries whose IDs are
            // special-token IDs.  tiktoken-format files often include special
            // tokens in the flat vocabulary; keeping them in the span map
            // conflicts with the separate SpecialVocab.
            let special_ids: std::collections::HashSet<u32> = spanning
                .specials()
                .span_map()
                .values()
                .copied()
                .collect();
            let raw_span_map = wc::load_base64_span_map_path::<u32, _>(&path)
                .map_err(to_pyerr)?;
            let filtered: wordchipper::vocab::SpanTokenMap<u32> = raw_span_map
                .into_iter()
                .filter(|(_, id)| !special_ids.contains(id))
                .collect();
            let span_vocab = wc::SpanMapVocab::from_span_map(filtered);
            let vocab = wc::UnifiedTokenVocab::from_span_vocab(spanning, span_vocab)
                .map_err(to_pyerr)?;
            let inner = options.inner().build(Arc::new(vocab));
            Ok(Tokenizer { inner })
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

    fn decode_bytes<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<u32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.inner
                    .try_decode_to_bytes(&tokens)
                    .and_then(|r| r.try_result())
            })
            .map_err(to_pyerr)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn decode_bytes_batch<'py>(
        &self,
        py: Python<'py>,
        batch: Vec<Vec<u32>>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let results = py
            .detach(|| {
                let refs = wc::inner_slice_view(&batch);
                self.inner
                    .try_decode_batch_to_bytes(&refs)
                    .and_then(|r| r.try_results())
            })
            .map_err(to_pyerr)?;
        Ok(results.iter().map(|b| PyBytes::new(py, b)).collect())
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
