use std::{hint::black_box, sync::LazyLock};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use wordchipper::encoders::token_span_encoder::SpanEncoderSelector;
use wordchipper_data::dataset::DatasetCacheConfig;

const BATCH_SIZE: usize = 1024;

struct Batch {
    samples: Vec<String>,
    total_bytes: usize,
}

impl Batch {
    fn strs(&self) -> Vec<&str> {
        self.samples.iter().map(|s| s.as_str()).collect()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

fn load_batch() -> Batch {
    let mut cache = DatasetCacheConfig::default()
        .init()
        .expect("failed to initialize dataset cache");

    let reader = cache
        .read_batches(0, true)
        .expect("failed to read dataset batches");

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
                    total_bytes: total_bytes,
                };
            }
        }
    }

    let total_bytes = samples.iter().map(|s| s.len()).sum();
    Batch {
        samples,
        total_bytes: total_bytes,
    }
}

static BATCH: LazyLock<Batch> = LazyLock::new(load_batch);

fn rayon_pool_size() -> usize {
    use rayon::prelude::*;
    let _ = [1, 2, 3].par_iter().max().unwrap().to_owned();

    rayon::current_num_threads()
}

fn bench_encoders(c: &mut Criterion) {
    const MODELS: &[(&str, bool)] = &[
        //        "openai::gpt2",
        //        "openai::p50k_base",
        ("openai::cl100k_base", true),
        //       "openai::o200k_base",
    ];

    const SPAN_ENCODERS: &[SpanEncoderSelector] = &[
        SpanEncoderSelector::BufferSweep,
        SpanEncoderSelector::MergeHeap,
        SpanEncoderSelector::PriorityMerge,
        SpanEncoderSelector::TailSweep,
    ];

    let max_pool = rayon_pool_size();
    let mut par_sizes: Vec<usize> = vec![max_pool];
    loop {
        let last = par_sizes.last().cloned().unwrap();
        let next = last / 2;
        if next < 2 {
            break;
        }
        par_sizes.push(next);
    }

    let strs = BATCH.strs();

    for (model, has_accel) in MODELS {
        let vocab = wordchipper_bench::load_bench_vocab(model);
        let accel_options: &[bool] = if *has_accel { &[false, true] } else { &[false] };

        let mut group = c.benchmark_group(format!("TokenEncoders/model={model}"));

        group.throughput(Throughput::Bytes(BATCH.total_bytes() as u64));
        group.sample_size(10);
        group.nresamples(1001);

        let options = wordchipper::TokenEncoderOptions::default().with_parallel(true);

        for &thread_count in par_sizes.iter() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .unwrap();

            for span_encoder in SPAN_ENCODERS {
                let options = options.clone().with_span_encoder(span_encoder.clone());

                for accel in accel_options {
                    let accel = *accel;
                    let label = format!("threads={thread_count};algo={span_encoder};accel={accel}");

                    let encoder = options
                        .clone()
                        .with_accelerated_lexers(accel)
                        .build(vocab.clone());
                    group.bench_function(label, |b| {
                        pool.install(|| {
                            b.iter(|| encoder.try_encode_batch(black_box(&strs)).unwrap())
                        })
                    });
                }
            }

            drop(pool);
        }
    }
}

criterion_group!(benches, bench_encoders);
criterion_main!(benches);
