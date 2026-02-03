//! # DataGym Vocabulary

use crate::types::CommonHashMap;
use serde_json::Value;

/// Parse a data gym "vocab.bpe" file from contents.
///
/// Handle extended ascii (https://en.wikipedia.org/wiki/Extended_ASCII)
/// Assume ISO/IEC 8859-1 (https://en.wikipedia.org/wiki/ISO/IEC_8859-1)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn parse_data_gym_vocab_bpe(
    vocab_bpe_contents: &str,
) -> (CommonHashMap<char, u8>, CommonHashMap<Vec<u8>, usize>) {
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

    // add the single byte tokens
    let mut bpe_ranks: CommonHashMap<Vec<u8>, usize> = rank_to_intbyte
        .into_iter()
        .enumerate()
        .map(|(i, b)| (vec![b], i))
        .collect();

    // vocab_bpe contains the merges along with associated ranks
    let bpe_merges: Vec<(&str, &str)> = vocab_bpe_contents
        .lines()
        .skip(1)
        .map(|line| line.split_once(' '))
        .filter(|item| item.is_some())
        .flatten()
        .collect();

    let mut n = bpe_ranks.len();
    for (first, second) in bpe_merges {
        let mut key = decode_data_gym(first, &data_gym_byte_to_byte);
        key.extend(decode_data_gym(second, &data_gym_byte_to_byte));
        bpe_ranks.insert(key, n);
        n += 1
    }

    (data_gym_byte_to_byte, bpe_ranks)
}

/// Parse a data gym "encoder.json" file from contents.
pub fn data_gym_parse_encoder_json(
    encoder_json_contents: &str,
    data_gym_byte_to_byte: CommonHashMap<char, u8>,
) -> CommonHashMap<Vec<u8>, usize> {

    /*
    # check that the encoder file matches the merges file
    # this sanity check is important since tiktoken assumes that ranks are ordered the same
    # as merge priority
    encoder_json = json.loads(read_file_cached(encoder_json_file, encoder_json_hash))
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    # drop these two special tokens if present, since they're not mergeable bpe tokens
    encoder_json_loaded.pop(b"<|endoftext|>", None)
    encoder_json_loaded.pop(b"<|startoftext|>", None)
    */

    // check that the encoder file matches the merges file
    // this sanity check is important since tiktoken assumes that ranks are ordered the same
    // as merge priority
    let encoder_json: Value =
        serde_json::from_str(&encoder_json_contents).unwrap_or(Value::Object(serde_json::Map::default()));
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

    encoder_json_loaded
}

/// Handle extended ascii (https://en.wikipedia.org/wiki/Extended_ASCII)
/// Assume ISO/IEC 8859-1 (https://en.wikipedia.org/wiki/ISO/IEC_8859-1)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_contents: &str,
    encoder_json_contents: &str,
    clobber_one_byte_tokens: bool,
) -> CommonHashMap<Vec<u8>, usize> {
    let (data_gym_byte_to_byte, mut bpe_ranks) = parse_data_gym_vocab_bpe(vocab_bpe_contents);

    let encoder_json_loaded = data_gym_parse_encoder_json(encoder_json_contents, data_gym_byte_to_byte);
    if clobber_one_byte_tokens {
        for (k, v) in &encoder_json_loaded {
            if k.len() == 1 {
                bpe_ranks.insert(k.clone(), *v);
            }
        }
    }

    assert_eq!(bpe_ranks.len(), encoder_json_loaded.len());

    bpe_ranks
}

fn decode_data_gym(value: &str, dict: &CommonHashMap<char, u8>) -> Vec<u8> {
    value
        .chars()
        .map(|c| dict.get(&c).copied().unwrap())
        .collect()
}