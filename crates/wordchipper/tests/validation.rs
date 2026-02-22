#![allow(missing_docs)]
#![cfg(feature = "client")]

use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;
use wordchipper::{
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    pretrained::openai::OATokenizer,
};

const SAMPLES: &[&str] = &[
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "It's a beautiful day, and I'll be taking my 3 dogs for a walk.",
    "Don't forget: the temperature is 72 degrees!",
    "  multiple   spaces  ",
    "line1\nline2\r\nline3",
    "123 + 456 = 789",
    "caf\u{00e9} na\u{00ef}ve \u{4f60}\u{597d}",
    "Geburtstag 2024: Alles Gute!",
    "$$$!!!...---",
    " ",
    "a",
    "\t\ttabs\tand\tspaces ",
    "emoji: \u{1f600}\u{1f680}\u{1f4a1}",
    "mixed: hello\u{00a0}world\u{2003}wide",
];

fn roundtrip_validation(model: OATokenizer) {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache).unwrap();
    let encoder = vocab.to_default_encoder();
    let decoder = vocab.to_default_decoder();

    for text in SAMPLES {
        let tokens = encoder.try_encode(text).unwrap();
        let decoded = decoder.try_decode_to_string(&tokens).unwrap();
        assert_eq!(
            &decoded.value, text,
            "Roundtrip mismatch for {model:?}: {text:?}"
        );
    }
}

fn tiktoken_validation(
    model: OATokenizer,
    tiktoken_bpe: &CoreBPE,
) {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache).unwrap();
    let encoder = vocab.to_default_encoder();

    for text in SAMPLES {
        let wc_tokens = encoder.try_encode(text).unwrap();
        let tt_tokens: Vec<u32> = tiktoken_bpe
            .encode_with_special_tokens(text)
            .into_iter()
            .map(|t| t as u32)
            .collect();

        assert_eq!(
            wc_tokens, tt_tokens,
            "Encode mismatch (wordchipper vs tiktoken) for {model:?}: {text:?}"
        );
    }
}

fn tokenizers_validation(
    model: OATokenizer,
    hf_tok: &Tokenizer,
) {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache).unwrap();
    let encoder = vocab.to_default_encoder();

    for text in SAMPLES {
        let wc_tokens = encoder.try_encode(text).unwrap();
        let hf_encoding = hf_tok.encode(*text, true).unwrap();
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();

        assert_eq!(
            wc_tokens, hf_tokens,
            "Encode mismatch (wordchipper vs tokenizers) for {model:?}: {text:?}"
        );
    }
}

#[test]
#[ignore]
fn cl100k_roundtrip() {
    roundtrip_validation(OATokenizer::Cl100kBase);
}

#[test]
#[ignore]
fn o200k_roundtrip() {
    roundtrip_validation(OATokenizer::O200kBase);
}

#[test]
#[ignore]
fn cl100k_vs_tiktoken() {
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    tiktoken_validation(OATokenizer::Cl100kBase, &bpe);
}

#[test]
#[ignore]
fn o200k_vs_tiktoken() {
    let bpe = tiktoken_rs::o200k_base().unwrap();
    tiktoken_validation(OATokenizer::O200kBase, &bpe);
}

#[test]
#[ignore]
fn cl100k_vs_tokenizers() {
    let tok = Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap();
    tokenizers_validation(OATokenizer::Cl100kBase, &tok);
}

#[test]
#[ignore]
fn o200k_vs_tokenizers() {
    let tok = Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap();
    tokenizers_validation(OATokenizer::O200kBase, &tok);
}
