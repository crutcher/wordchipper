use std::sync::Arc;

use pyo3::{
    exceptions::{PyIOError, PyValueError},
    prelude::*,
};
use wordchipper::{
    TokenDecoder,
    TokenDecoderBuilder,
    TokenEncoder,
    TokenEncoderBuilder,
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
    vocab: Arc<UnifiedTokenVocab<u32>>,
    encoder: Arc<dyn TokenEncoder<u32>>,
    decoder: Arc<dyn TokenDecoder<u32>>,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn from_pretrained(name: &str) -> PyResult<Self> {
        let mut disk_cache = WordchipperDiskCache::default();
        let vocab: Arc<UnifiedTokenVocab<u32>> = wordchipper::get_model(name, &mut disk_cache)
            .map_err(to_pyerr)?
            .into();
        let encoder = TokenEncoderBuilder::default(vocab.clone());
        let decoder = TokenDecoderBuilder::default(vocab.clone());

        Ok(Tokenizer {
            vocab,
            encoder,
            decoder,
        })
    }

    fn encode(
        &self,
        text: &str,
    ) -> PyResult<Vec<u32>> {
        self.encoder.try_encode(text).map_err(to_pyerr)
    }

    fn encode_batch(
        &self,
        texts: Vec<String>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let refs = inner_str_view(&texts);
        self.encoder.try_encode_batch(&refs).map_err(to_pyerr)
    }

    fn decode(
        &self,
        tokens: Vec<u32>,
    ) -> PyResult<String> {
        self.decoder
            .try_decode_to_string(&tokens)
            .map_err(to_pyerr)?
            .try_result()
            .map_err(to_pyerr)
    }

    fn decode_batch(
        &self,
        batch: Vec<Vec<u32>>,
    ) -> PyResult<Vec<String>> {
        let refs = inner_slice_view(&batch);
        self.decoder
            .try_decode_batch_to_strings(&refs)
            .map_err(to_pyerr)?
            .try_results()
            .map_err(to_pyerr)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    #[getter]
    fn max_token(&self) -> Option<u32> {
        self.vocab.max_token()
    }

    fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.vocab.lookup_token(token.as_bytes())
    }

    fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.vocab
            .unified_dictionary()
            .get(&id)
            .map(|bytes| string_from_utf8_lossy(bytes.clone()))
    }

    fn get_special_tokens(&self) -> Vec<(String, u32)> {
        self.vocab
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
        path: &str,
    ) -> PyResult<()> {
        save_base64_span_map_path(self.vocab.span_vocab().span_map(), path).map_err(to_pyerr)
    }
}

#[pymodule]
fn _wordchipper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
