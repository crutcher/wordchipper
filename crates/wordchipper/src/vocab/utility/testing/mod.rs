//! # Vocab Testing Tools

use crate::spanner::TextSpanConfig;
use crate::types::TokenType;
use crate::vocab::vocab_types::SpanTokenMap;
use crate::vocab::{ByteMapVocab, SpanMapVocab, UnifiedTokenVocab};

/// Create a test [`UnifiedTokenVocab`].
pub fn build_test_vocab<T: TokenType, C>(
    byte_vocab: ByteMapVocab<T>,
    segmentation: C,
) -> UnifiedTokenVocab<T>
where
    C: Into<TextSpanConfig<T>>,
{
    let mut span_map: SpanTokenMap<T> = Default::default();
    span_map.extend(
        [
            ("at", 300),
            ("ate", 301),
            ("th", 302),
            ("the", 303),
            (", ", 304),
            ("on", 305),
            ("he", 306),
            ("ll", 307),
            ("hell", 308),
            ("hello", 309),
            ("wo", 310),
            ("ld", 311),
            ("rld", 312),
            ("world", 313),
            ("fo", 314),
            ("for", 315),
            ("all", 316),
            (". ", 317),
        ]
        .into_iter()
        .map(|(k, v)| (k.as_bytes().to_vec(), T::from_usize(v).unwrap())),
    );

    let span_vocab = SpanMapVocab::init(byte_vocab, span_map).unwrap();

    UnifiedTokenVocab::from_span_vocab(segmentation.into(), span_vocab)
}
