#![allow(missing_docs)]

use std::sync::{Arc, LazyLock};

use ::tokenizers::Tokenizer;
use arrow::array::Array;
use divan::{Bencher, black_box, counter::BytesCount};
use tiktoken_rs::CoreBPE;
use wordchipper::{
    TokenEncoder,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    encoders::span_encoders::{
        IncrementalSweepSpanEncoder,
        MergeHeapSpanEncoder,
        PriorityMergeSpanEncoder,
        SpanEncoder,
        TokenSpanEncoder,
    },
    pretrained::openai::OATokenizer,
    spanning::TextSpannerBuilder,
    support::concurrency::rayon::ParallelRayonEncoder,
};
use wordchipper_data::dataset::DatasetCacheConfig;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const BATCH_SIZE: usize = 1024;

struct Batch {
    samples: Vec<String>,
    total_bytes: usize,
}

impl Batch {
    fn strs(&self) -> Vec<&str> {
        self.samples.iter().map(|s| s.as_str()).collect()
    }
}

fn load_batch() -> Batch {
    let data_dir = std::env::var("WORDCHIPPER_BENCH_DATA")
        .unwrap_or_else(|_| "/tmp/wordchipper-bench-data".to_string());

    let cache = DatasetCacheConfig::default()
        .with_cache_dir(data_dir.clone())
        .init()
        .unwrap_or_else(|e| panic!("Failed to init dataset cache at {data_dir}: {e}"));

    let reader = cache.read_cached_batches(0).unwrap_or_else(|e| {
        panic!(
            "Failed to read shard 0 from {data_dir}. \
                 Download it first with sample-timer or set WORDCHIPPER_BENCH_DATA: {e}"
        )
    });

    let mut samples = Vec::with_capacity(BATCH_SIZE);
    for batch in reader {
        let batch = batch.unwrap();
        let column = batch
            .column_by_name("text")
            .expect("missing 'text' column")
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        for val in column {
            samples.push(val.unwrap().to_string());
            if samples.len() >= BATCH_SIZE {
                let total_bytes = samples.iter().map(|s| s.len()).sum();
                return Batch {
                    samples,
                    total_bytes,
                };
            }
        }
    }

    let total_bytes = samples.iter().map(|s| s.len()).sum();
    Batch {
        samples,
        total_bytes,
    }
}

static BATCH: LazyLock<Batch> = LazyLock::new(load_batch);

fn load_wc_variant(
    model: OATokenizer,
    se_builder: Arc<dyn Fn() -> Box<dyn SpanEncoder<u32>> + Send + Sync>,
) -> Arc<dyn TokenEncoder<u32>> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: Arc<UnifiedTokenVocab<u32>> =
        model.load_vocab::<u32>(&mut disk_cache).unwrap().into();
    let spanner = TextSpannerBuilder::from_vocab(&vocab).build();
    let inner: Arc<dyn TokenEncoder<u32>> =
        Arc::new(TokenSpanEncoder::<u32>::new(spanner, vocab, se_builder));
    Arc::new(ParallelRayonEncoder::new(inner))
}

// cl100k variants
static WC_SWEEP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::Cl100kBase,
        Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<u32>::default())),
    )
});
static WC_HEAP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::Cl100kBase,
        Arc::new(|| Box::new(MergeHeapSpanEncoder::<u32>::default())),
    )
});
static WC_PMERGE_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::Cl100kBase,
        Arc::new(|| Box::new(PriorityMergeSpanEncoder::<u32>::default())),
    )
});

// o200k variants
static WC_SWEEP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::O200kBase,
        Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<u32>::default())),
    )
});
static WC_HEAP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::O200kBase,
        Arc::new(|| Box::new(MergeHeapSpanEncoder::<u32>::default())),
    )
});
static WC_PMERGE_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::O200kBase,
        Arc::new(|| Box::new(PriorityMergeSpanEncoder::<u32>::default())),
    )
});

static TT_CL100K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::cl100k_base().unwrap()));
static TT_O200K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::o200k_base().unwrap()));

static HF_CL100K: LazyLock<Arc<Tokenizer>> = LazyLock::new(|| {
    Arc::new(Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap())
});
static HF_O200K: LazyLock<Arc<Tokenizer>> =
    LazyLock::new(|| Arc::new(Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap()));

mod incremental_sweep {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let strs = BATCH.strs();
        let encoder = &*WC_SWEEP_CL100K;
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let strs = BATCH.strs();
        let encoder = &*WC_SWEEP_O200K;
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }
}

mod merge_heap {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let strs = BATCH.strs();
        let encoder = &*WC_HEAP_CL100K;
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let strs = BATCH.strs();
        let encoder = &*WC_HEAP_O200K;
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }
}

mod priority_merge {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let strs = BATCH.strs();
        let encoder = &*WC_PMERGE_CL100K;
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let strs = BATCH.strs();
        let encoder = &*WC_PMERGE_O200K;
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }
}

mod tiktoken {
    use rayon::prelude::*;

    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let bpe = &*TT_CL100K;
        let strs = BATCH.strs();
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| {
                strs.par_iter()
                    .map(|s| bpe.encode_with_special_tokens(s))
                    .collect::<Vec<_>>()
            });
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let bpe = &*TT_O200K;
        let strs = BATCH.strs();
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| {
                strs.par_iter()
                    .map(|s| bpe.encode_with_special_tokens(s))
                    .collect::<Vec<_>>()
            });
    }
}

mod tokenizers {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let tok = &*HF_CL100K;
        let strs = BATCH.strs();
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| tok.encode_batch(black_box(strs.clone()), true).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let tok = &*HF_O200K;
        let strs = BATCH.strs();
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| tok.encode_batch(black_box(strs.clone()), true).unwrap());
    }
}
