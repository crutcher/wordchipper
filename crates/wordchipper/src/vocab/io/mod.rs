//! # Vocabulary IO

pub mod base64_vocab;
pub mod tiktoken_io;

pub use base64_vocab::{
    load_base64_span_map_path, read_base64_span_map, save_base64_span_map_path,
    write_base64_span_map,
};
pub use tiktoken_io::{
    load_tiktoken_vocab_path, read_tiktoken_vocab, save_tiktoken_vocab_path, write_tiktoken_vocab,
};
