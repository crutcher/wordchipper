#![allow(missing_docs)]

use std::sync::{Arc, LazyLock};

use divan::{Bencher, black_box, counter::BytesCount};
use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;
use wordchipper::{
    TokenEncoder,
    encoders::token_span_encoder::SpanEncoderSelector,
    pretrained::openai::OATokenizer,
};

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

fn load_wc_variant(
    model: OATokenizer,
    selector: SpanEncoderSelector,
) -> Arc<dyn TokenEncoder<u32>> {
    wordchipper_bench::encoder_builder(model, selector)
        .with_parallel(false)
        .build()
}

// BufferSwee
static WC_BSWEEP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::Cl100kBase, SpanEncoderSelector::BufferSweep));
static WC_BSWEEP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::O200kBase, SpanEncoderSelector::BufferSweep));

// TailSweep
static WC_TSWEEP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::Cl100kBase, SpanEncoderSelector::TailSweep));
static WC_TSWEEP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::O200kBase, SpanEncoderSelector::TailSweep));

// MergeHeap
static WC_HEAP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::Cl100kBase, SpanEncoderSelector::MergeHeap));
static WC_HEAP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::O200kBase, SpanEncoderSelector::MergeHeap));

// PriorityMerge
static WC_PMERGE_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::Cl100kBase, SpanEncoderSelector::PriorityMerge));
static WC_PMERGE_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_variant(OATokenizer::O200kBase, SpanEncoderSelector::PriorityMerge));

struct TiktokenFixture {
    bpe: Arc<CoreBPE>,
}

static TT_CL100K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::cl100k_base().unwrap()),
});

static TT_O200K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::o200k_base().unwrap()),
});

static HF_CL100K: LazyLock<Arc<Tokenizer>> = LazyLock::new(|| {
    Arc::new(Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap())
});

static HF_O200K: LazyLock<Arc<Tokenizer>> =
    LazyLock::new(|| Arc::new(Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap()));

mod english {
    use super::*;

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_BSWEEP_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_BSWEEP_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_TSWEEP_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_TSWEEP_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_HEAP_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_HEAP_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_PMERGE_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_PMERGE_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let bpe = &TT_CL100K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let bpe = &TT_O200K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let tok = &*HF_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let tok = &*HF_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }
    }
}

mod diverse {
    use super::*;

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_BSWEEP_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_BSWEEP_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_TSWEEP_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_TSWEEP_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_HEAP_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_HEAP_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_PMERGE_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_PMERGE_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let bpe = &TT_CL100K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let bpe = &TT_O200K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let tok = &*HF_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let tok = &*HF_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }
    }
}
