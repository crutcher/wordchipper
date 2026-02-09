//! # Tiktoken Vocabulary IO

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::Context;
use base64::{Engine, prelude::BASE64_STANDARD};

use crate::{
    types::TokenType,
    vocab::{SpanMapVocab, vocab_types::SpanTokenMap},
};

/// Load a [`SpanMapVocab`] from a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_base64_span_vocab_path<T, P>(path: P) -> anyhow::Result<SpanMapVocab<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    Ok(load_base64_span_map_path(path)?.into())
}

/// Load a [`SpanTokenMap`] from a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_base64_span_map_path<T, P>(path: P) -> anyhow::Result<SpanTokenMap<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    read_base64_span_map(reader)
}

/// Read a [`SpanTokenMap`] from a base64 vocab line reader.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `span_map` - the vocabulary to extend.
/// * `reader` - the line reader.
pub fn read_base64_span_map<T, R>(reader: R) -> anyhow::Result<SpanTokenMap<T>>
where
    T: TokenType,
    R: BufRead,
{
    let mut vocab = SpanTokenMap::default();

    let stream = reader.lines();
    for line in stream {
        let line = line?;
        let s: &str = line.as_ref();

        let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
        assert_eq!(parts.len(), 2);

        let span = BASE64_STANDARD.decode(parts[0])?;

        let token = T::from_u64(parts[1].parse()?).context("token out of range")?;

        vocab.insert(span, token);
    }

    Ok(vocab)
}

/// Save a [`SpanTokenMap`] to a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `path` - the path to save the vocabulary to.
pub fn save_base64_span_map_path<T: TokenType, P: AsRef<Path>>(
    span_map: &SpanTokenMap<T>,
    path: P,
) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    write_base64_span_map(span_map, &mut writer)
}

/// Write a [`SpanTokenMap`] to a [`Write`] writer.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `writer` - the writer to target.
pub fn write_base64_span_map<T, W>(
    span_map: &SpanTokenMap<T>,
    writer: &mut W,
) -> anyhow::Result<()>
where
    T: TokenType,
    W: Write,
{
    let mut items: Vec<(T, &Vec<u8>)> = span_map
        .iter()
        .map(|(chunk, &token)| (token, chunk))
        .collect();
    items.sort_by_key(|(t, _)| *t);

    for (token, chunk) in items {
        writeln!(
            writer,
            "{} {}",
            BASE64_STANDARD.encode(chunk),
            token.to_u64().unwrap()
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_tiktoken() {
        type T = u32;

        let mut span_map: SpanTokenMap<T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                save_base64_span_map_path(&span_map, &path).expect("Failed to save vocab");

                let loaded_vocab = load_base64_span_map_path(&path).expect("Failed to load vocab");

                assert_eq!(&loaded_vocab, &span_map);

                Ok(())
            })
            .unwrap();
    }
}
