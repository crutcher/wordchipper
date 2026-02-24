use std::sync::Arc;

use pyo3::{
    exceptions::{PyIOError, PyValueError},
    prelude::*,
};
use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    Tokenizer as _Tokenizer,
    TokenizerOptions,
    UnifiedTokenVocab,
    VocabIndex,
    WCError,
    disk_cache::WordchipperDiskCache,
    support::{
        slices::{inner_slice_view, inner_str_view},
        strings::string_from_utf8_lossy,
    },
    vocab::io::save_base64_span_map_path,
};

fn to_pyerr(err: WCError) -> PyErr {
    match err {
        WCError::Io(e) => PyIOError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

#[pyclass]
struct Tokenizer {
    inner: Arc<_Tokenizer<u32>>,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn from_pretrained(
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Self> {
        py.detach(|| {
            let mut disk_cache = WordchipperDiskCache::default();
            let vocab: Arc<UnifiedTokenVocab<u32>> = wordchipper::get_model(name, &mut disk_cache)
                .map_err(to_pyerr)?
                .into();

            let inner = TokenizerOptions::default()
                .with_parallel(true)
                .build(vocab.clone());

            Ok(Tokenizer { inner })
        })
    }

    fn encode(
        &self,
        py: Python<'_>,
        text: String,
    ) -> PyResult<Vec<u32>> {
        let inner = self.inner.clone();
        py.detach(move || inner.try_encode(&text)).map_err(to_pyerr)
    }

    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let inner = self.inner.clone();
        py.detach(move || {
            let refs = inner_str_view(&texts);
            inner.try_encode_batch(&refs)
        })
        .map_err(to_pyerr)
    }

    fn decode(
        &self,
        py: Python<'_>,
        tokens: Vec<u32>,
    ) -> PyResult<String> {
        let inner = self.inner.clone();
        py.detach(move || {
            inner
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
        let inner = self.inner.clone();
        py.detach(move || {
            let refs = inner_slice_view(&batch);
            inner
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
            .map(|bytes| string_from_utf8_lossy(bytes.clone()))
    }

    fn get_special_tokens(&self) -> Vec<(String, u32)> {
        self.inner
            .vocab()
            .special_vocab()
            .span_map()
            .iter()
            .map(|(bytes, &token)| (string_from_utf8_lossy(bytes.to_vec()), token))
            .collect()
    }

    #[staticmethod]
    fn available_models() -> Vec<String> {
        wordchipper::list_models(false)
    }

    fn save_base64_vocab(
        &self,
        py: Python<'_>,
        path: &str,
    ) -> PyResult<()> {
        py.detach(|| {
            save_base64_span_map_path(self.inner.vocab().span_vocab().span_map(), path)
                .map_err(to_pyerr)
        })
    }
}

#[pymodule]
fn _wordchipper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
