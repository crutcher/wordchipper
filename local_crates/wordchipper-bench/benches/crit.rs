use std::{hint::black_box, sync::LazyLock};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use wordchipper::pretrained::openai::OATokenizer;
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

fn bench_encoders(c: &mut Criterion) {
    let strs = BATCH.strs();

    for parallel in [false, true] {
        for model in [OATokenizer::Cl100kBase, OATokenizer::O200kBase] {
            let vocab = wordchipper_bench::load_vocab::<u32>(model);

            let group_name = format!("par={}/{}", parallel, model);
            let mut group = c.benchmark_group(&group_name);

            group.throughput(Throughput::Bytes(BATCH.total_bytes() as u64));

            group.sample_size(10);
            group.nresamples(1001);

            for accel in [false, true] {
                let encoder = wordchipper::TokenEncoderOptions::default()
                    .with_parallel(parallel)
                    .with_accelerated_lexers(accel)
                    .build(vocab.clone());

                group.bench_function(&format!("accel={}", accel), |b| {
                    b.iter(|| encoder.try_encode_batch(black_box(&strs)).unwrap())
                });
            }
        }
    }
}

criterion_group!(benches, bench_encoders);
criterion_main!(benches);
