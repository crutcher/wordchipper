//! # `DataGym` Vocabulary

use crate::types::CommonHashMap;
use serde_json::Value;
use std::io::{BufRead, BufReader};

fn data_gym_default_maps() -> (Vec<u8>, CommonHashMap<char, u8>) {
    let mut rank_to_intbyte: Vec<u8> = vec![];
    rank_to_intbyte.extend(0x21..=0x7E);
    rank_to_intbyte.extend(0xA1..0xAD);
    rank_to_intbyte.extend(0xAE..=0xFF);

    let mut data_gym_byte_to_byte: CommonHashMap<char, u8> = rank_to_intbyte
        .iter()
        .map(|&b| (char::from(b), b))
        .collect();
    let mut n = 0u32;
    for b in 0..=255 {
        if !rank_to_intbyte.contains(&b) {
            rank_to_intbyte.push(b);
            data_gym_byte_to_byte.insert(char::from_u32(256 + n).unwrap(), b);
            n += 1;
        }
    }
    assert_eq!(rank_to_intbyte.len(), 256);

    (rank_to_intbyte, data_gym_byte_to_byte)
}

fn data_gym_blank_ranks() -> (CommonHashMap<char, u8>, CommonHashMap<Vec<u8>, usize>) {
    let (rank_to_intbyte, data_gym_byte_to_byte) = data_gym_default_maps();

    // add the single byte tokens
    let bpe_ranks: CommonHashMap<Vec<u8>, usize> = rank_to_intbyte
        .into_iter()
        .enumerate()
        .map(|(i, b)| (vec![b], i))
        .collect();

    (data_gym_byte_to_byte, bpe_ranks)
}

/// Parse a data gym "vocab.bpe" file from contents.
///
/// Handle extended ascii (<https://en.wikipedia.org/wiki/Extended_ASCII>)
/// Assume ISO/IEC 8859-1 (<https://en.wikipedia.org/wiki/ISO/IEC_8859-1>)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn parse_vocab_bpe(
    vocab_bpe_contents: &str
) -> anyhow::Result<(CommonHashMap<char, u8>, CommonHashMap<Vec<u8>, usize>)> {
    let reader = BufReader::new(vocab_bpe_contents.as_bytes());
    read_vocab_bpe(reader)
}

/// Read a daty gym "vocab.bpe" file.
pub fn read_vocab_bpe<R>(
    vocab_bpe_reader: R
) -> anyhow::Result<(CommonHashMap<char, u8>, CommonHashMap<Vec<u8>, usize>)>
where
    R: BufRead,
{
    let (data_gym_byte_to_byte, mut bpe_ranks) = data_gym_blank_ranks();

    let mut bpe_merges: Vec<(String, String)> = vec![];
    for line in vocab_bpe_reader.lines().skip(1) {
        let line = line?;

        if let Some((first, second)) = line.split_once(' ') {
            bpe_merges.push((first.to_string(), second.to_string()))
        }
    }

    let mut n = bpe_ranks.len();
    for (first, second) in bpe_merges {
        let mut key = decode_data_gym(first.as_str(), &data_gym_byte_to_byte);
        key.extend(decode_data_gym(second.as_str(), &data_gym_byte_to_byte));
        bpe_ranks.insert(key, n);
        n += 1
    }

    Ok((data_gym_byte_to_byte, bpe_ranks))
}

/// Parse a data gym "encoder.json" file from contents.
pub fn parse_encoder_json(
    encoder_json_contents: &str,
    data_gym_byte_to_byte: CommonHashMap<char, u8>,
) -> anyhow::Result<CommonHashMap<Vec<u8>, usize>> {
    let reader = BufReader::new(encoder_json_contents.as_bytes());
    read_encoder_json(reader, data_gym_byte_to_byte)
}

/// Parse a data gym "encoder.json" file from contents.
pub fn read_encoder_json<R>(
    encoder_json_reader: R,
    data_gym_byte_to_byte: CommonHashMap<char, u8>,
) -> anyhow::Result<CommonHashMap<Vec<u8>, usize>>
where
    R: BufRead,
{
    // check that the encoder file matches the merges file
    // this sanity check is important since tiktoken assumes that ranks are ordered the same
    // as merge priority
    let encoder_json: Value = serde_json::from_reader(encoder_json_reader)
        .unwrap_or(Value::Object(serde_json::Map::default()));
    let mut encoder_json_loaded: CommonHashMap<Vec<u8>, usize> = encoder_json
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, val)| {
            (
                decode_data_gym(key, &data_gym_byte_to_byte),
                val.as_u64().unwrap() as usize,
            )
        })
        .collect();
    encoder_json_loaded.remove("<|endoftext|>".as_bytes());
    encoder_json_loaded.remove("<|endoftext|>".as_bytes());

    Ok(encoder_json_loaded)
}

/// Handle extended ascii (<https://en.wikipedia.org/wiki/Extended_ASCII>)
/// Assume ISO/IEC 8859-1 (<https://en.wikipedia.org/wiki/ISO/IEC_8859-1>)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn parse_data_gym(
    vocab_bpe_contents: &str,
    encoder_json_contents: &str,
    clobber_one_byte_tokens: bool,
) -> anyhow::Result<CommonHashMap<Vec<u8>, usize>> {
    let v_reader = BufReader::new(vocab_bpe_contents.as_bytes());
    let e_reader = BufReader::new(encoder_json_contents.as_bytes());

    read_data_gym(v_reader, e_reader, clobber_one_byte_tokens)
}

/// Handle extended ascii (<https://en.wikipedia.org/wiki/Extended_ASCII>)
/// Assume ISO/IEC 8859-1 (<https://en.wikipedia.org/wiki/ISO/IEC_8859-1>)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn read_data_gym<VR, ER>(
    vocab_bpe_reader: VR,
    encoder_json_reader: ER,
    clobber_one_byte_tokens: bool,
) -> anyhow::Result<CommonHashMap<Vec<u8>, usize>>
where
    VR: BufRead,
    ER: BufRead,
{
    let (data_gym_byte_to_byte, mut bpe_ranks) = read_vocab_bpe(vocab_bpe_reader)?;

    let encoder_json_loaded = read_encoder_json(encoder_json_reader, data_gym_byte_to_byte)?;
    if clobber_one_byte_tokens {
        for (k, v) in &encoder_json_loaded {
            if k.len() == 1 {
                bpe_ranks.insert(k.clone(), *v);
            }
        }
    }

    assert_eq!(bpe_ranks.len(), encoder_json_loaded.len());

    Ok(bpe_ranks)
}

fn decode_data_gym(
    value: &str,
    dict: &CommonHashMap<char, u8>,
) -> Vec<u8> {
    value
        .chars()
        .map(|c| dict.get(&c).copied().unwrap())
        .collect()
}
