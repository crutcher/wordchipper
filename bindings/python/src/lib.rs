use pyo3::{
    prelude::*,
    pymodule,
};

pub(crate) mod wc {
    pub use wordchipper::{
        SpecialFilter,
        Tokenizer,
        TokenizerOptions,
        disk_cache::WordchipperDiskCache,
        load_vocab,
        support::{
            slices::{
                inner_slice_view,
                inner_str_view,
            },
            strings::string_from_utf8_lossy,
        },
        vocab::io::save_base64_span_map_path,
    };
}

mod support;
mod tokenizer;
mod vocab;

#[pymodule]
fn _wordchipper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tokenizer::TokenizerOptions>()?;
    m.add_class::<tokenizer::_Tokenizer>()?;
    m.add_class::<vocab::SpecialFilter>()?;
    m.add_class::<vocab::_Vocab>()?;
    Ok(())
}
