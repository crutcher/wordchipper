//! # Tiktoken Vocabulary IO

use crate::types::{CommonHashMap, SpanTokenMap, TokenType};
use anyhow::Context;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Load a [`SpanTokenMap`] from a tiktoken vocab file.
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_span_map_from_tiktoken_path<T, P>(path: P) -> anyhow::Result<SpanTokenMap<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    load_span_map_from_tiktoken_reader(reader)
}

/// Update a [`SpanTokenMap`] from a tiktoken vocab [`BufRead`] stream.
///
/// # Arguments
/// * `span_map` - the vocabulary to extend.
/// * `reader` - the line reader.
pub fn load_span_map_from_tiktoken_reader<T, R>(reader: R) -> anyhow::Result<SpanTokenMap<T>>
where
    T: TokenType,
    R: BufRead,
{
    let mut vocab: CommonHashMap<Vec<u8>, T> = Default::default();

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

/// Save a [`SpanTokenMap`] to a tiktoken vocab file.
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `path` - the path to save the vocabulary to.
pub fn save_span_map_to_tiktoken_path<T: TokenType, P: AsRef<Path>>(
    span_map: &SpanTokenMap<T>,
    path: P,
) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    save_span_map_to_tiktoken_writer(span_map, &mut writer)
}

/// Save a [`SpanTokenMap`] to a [`Write`] writer.
pub fn save_span_map_to_tiktoken_writer<T, W>(
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

        let mut span_map: CommonHashMap<Vec<u8>, T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                save_span_map_to_tiktoken_path(&span_map, &path).expect("Failed to save vocab");

                let loaded_vocab =
                    load_span_map_from_tiktoken_path(&path).expect("Failed to load vocab");

                assert_eq!(&loaded_vocab, &span_map);

                Ok(())
            })
            .unwrap();
    }
}
