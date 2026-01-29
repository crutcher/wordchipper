//! # Tiktoken Vocabulary IO

use std::fs::File;
use crate::types::{SpanTokenMap, TokenType};
use crate::vocab::io::base64_vocab::{
    read_base64_span_map,
    write_base64_span_map,
};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Load a [`SpanTokenMap`] from a tiktoken vocab file.
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_tiktoken_vocab_path<T, P>(path: P) -> anyhow::Result<SpanTokenMap<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    read_tiktoken_vocab(reader)
}

/// Update a [`SpanTokenMap`] from a tiktoken vocab [`BufRead`] stream.
///
/// # Arguments
/// * `span_map` - the vocabulary to extend.
/// * `reader` - the line reader.
pub fn read_tiktoken_vocab<T, R>(reader: R) -> anyhow::Result<SpanTokenMap<T>>
where
    T: TokenType,
    R: BufRead,
{
    read_base64_span_map(reader)
}

/// Save a [`SpanTokenMap`] to a tiktoken vocab file.
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `path` - the path to save the vocabulary to.
pub fn save_tiktoken_vocab_path<T: TokenType, P: AsRef<Path>>(
    span_map: &SpanTokenMap<T>,
    path: P,
) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    write_tiktoken_vocab(span_map, &mut writer)
}

/// Save a [`SpanTokenMap`] to a [`Write`] writer.
pub fn write_tiktoken_vocab<T, W>(
    span_map: &SpanTokenMap<T>,
    writer: &mut W,
) -> anyhow::Result<()>
where
    T: TokenType,
    W: Write,
{
    write_base64_span_map(span_map, writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CommonHashMap;

    #[test]
    fn test_save_load_tiktoken() {
        type T = u32;

        let mut span_map: CommonHashMap<Vec<u8>, T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                save_tiktoken_vocab_path(&span_map, &path).expect("Failed to save vocab");

                let loaded_vocab = load_tiktoken_vocab_path(&path).expect("Failed to load vocab");

                assert_eq!(&loaded_vocab, &span_map);

                Ok(())
            })
            .unwrap();
    }
}
