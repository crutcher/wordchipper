//! # Tiktoken Vocabulary IO

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use base64::{Engine, prelude::BASE64_STANDARD};

use crate::{
    errors::WordchipperError,
    spanners::TextSpanningConfig,
    types::TokenType,
    vocab::{SpanMapVocab, UnifiedTokenVocab, vocab_types::SpanTokenMap},
};

/// Build a [`UnifiedTokenVocab`] from a pretrained bas64 vocab file.
///
/// ## Arguments
/// * `data_path` - path to the file.
/// * `pattern` - the word split pattern.
/// * `special_tokens` - the special tokens.
pub fn load_base64_unified_vocab_path<T: TokenType>(
    path: impl AsRef<Path>,
    spanning: TextSpanningConfig<T>,
) -> crate::errors::Result<UnifiedTokenVocab<T>> {
    let reader = BufReader::new(File::open(path)?);
    read_base64_unified_vocab(reader, spanning)
}

/// Build a [`UnifiedTokenVocab`] from a pretrained bas64 vocab file.
///
/// ## Arguments
/// * `data_path` - path to the file.
/// * `pattern` - the word split pattern.
/// * `special_tokens` - the special tokens.
pub fn read_base64_unified_vocab<T: TokenType, R: BufRead>(
    reader: R,
    spanning: TextSpanningConfig<T>,
) -> crate::errors::Result<UnifiedTokenVocab<T>> {
    UnifiedTokenVocab::from_span_vocab(spanning, read_base64_span_map(reader)?.into())
}

/// Load a [`SpanMapVocab`] from a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_base64_span_vocab_path<T, P>(path: P) -> crate::errors::Result<SpanMapVocab<T>>
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
pub fn load_base64_span_map_path<T, P>(path: P) -> crate::errors::Result<SpanTokenMap<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let reader = BufReader::new(File::open(path)?);
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
pub fn read_base64_span_map<T, R>(reader: R) -> crate::errors::Result<SpanTokenMap<T>>
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

        let span = BASE64_STANDARD
            .decode(parts[0])
            .map_err(|e| WordchipperError::Parse(e.to_string()))?;

        let id: u64 = parts[1]
            .parse()
            .map_err(|e: core::num::ParseIntError| WordchipperError::Parse(e.to_string()))?;
        let token = T::from_u64(id).ok_or(WordchipperError::TokenOutOfRange)?;

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
) -> crate::errors::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
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
) -> crate::errors::Result<()>
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
