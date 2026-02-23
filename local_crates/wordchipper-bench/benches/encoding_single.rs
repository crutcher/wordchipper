#![allow(missing_docs)]

use divan::{Bencher, black_box, counter::BytesCount};
use wordchipper::{
    TokenEncoderOptions,
    encoders::token_span_encoder::SpanEncoderSelector,
    pretrained::openai::OATokenizer,
};
use wordchipper_bench::{HF_CL100K, HF_O200K};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

static DIVERSE_CORPUS: &str = include_str!("data/multilingual.txt");
static ENGLISH_CORPUS: &str = include_str!("data/english.txt");

fn diverse_text() -> String {
    DIVERSE_CORPUS.repeat(10)
}

fn english_text() -> String {
    ENGLISH_CORPUS.repeat(10)
}

pub fn bench_wc(
    bencher: Bencher,
    text: &str,
    model: OATokenizer,
    selector: SpanEncoderSelector,
    accelerator: bool,
) {
    let encoder = wordchipper_bench::load_encoder::<u32>(
        model,
        TokenEncoderOptions::default()
            .with_accelerated_lexers(accelerator)
            .with_span_encoder(selector)
            .with_parallel(true)
            .with_concurrent(false),
    );

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| encoder.try_encode(black_box(text)).unwrap());
}

pub fn bench_tt(
    bencher: Bencher,
    text: &str,
    tok: &tiktoken_rs::CoreBPE,
) {
    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode_with_special_tokens(black_box(text)));
}

pub fn bench_hf(
    bencher: Bencher,
    text: &str,
    name: &str,
) {
    let tok = tokenizers::Tokenizer::from_pretrained(name, None).unwrap();

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode(black_box(text), true).unwrap());
}

mod english {
    use super::*;

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_tt(
                bencher,
                &english_text(),
                &tiktoken_rs::cl100k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_tt(
                bencher,
                &english_text(),
                &tiktoken_rs::o200k_base().unwrap(),
            )
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_hf(bencher, &english_text(), HF_CL100K)
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_hf(bencher, &english_text(), HF_O200K)
        }
    }
}

mod diverse {
    use super::*;

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }
    }

    mod merge_heap {
        use super::*;
        use crate::bench_wc;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::Cl100kBase,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OATokenizer::O200kBase,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_tt(
                bencher,
                &diverse_text(),
                &tiktoken_rs::cl100k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_tt(
                bencher,
                &diverse_text(),
                &tiktoken_rs::o200k_base().unwrap(),
            )
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_hf(bencher, &diverse_text(), HF_CL100K)
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_hf(bencher, &diverse_text(), HF_O200K)
        }
    }
}
