use std::{fs::DirEntry, path::Path, str::FromStr, time::Duration};

use wordchipper::encoders::token_span_encoder::SpanEncoderSelector;

fn main() {
    let data_dir = std::env::current_dir()
        .unwrap()
        .join("target")
        .join("criterion");

    build(&data_dir);
}

#[allow(unused)]
#[derive(Debug, serde::Deserialize)]
struct Record {
    group: String,
    function: String,
    value: u64,
    throughput_num: u64,
    throughput_type: String,
    sample_measured_value: f64,
    unit: String,
    iteration_count: u64,
}

pub fn build(data_dir: &Path) {
    for model_dir_read in data_dir.read_dir().unwrap() {
        let model_dir = model_dir_read.unwrap();
        let model_dir_name = model_dir.file_name().to_string_lossy().to_string();
        if model_dir_name.starts_with("TokenEncoder_") {
            let model_name = model_dir_name.split("_").nth(1).unwrap();
            println!("model={:?}", model_name);

            for seq_dir_read in model_dir.path().read_dir().unwrap() {
                let seq_dir = seq_dir_read.unwrap();
                let seq_name = seq_dir.file_name().to_string_lossy().to_string();

                let parts = seq_name.splitn(2, "_").collect::<Vec<_>>();
                if parts.len() != 2 {
                    continue;
                }
                let span_name = parts[0];
                let lexer_name = parts[1];

                if SpanEncoderSelector::from_str(&span_name).is_err() {
                    continue;
                }

                println!("{span_name}/{lexer_name}");

                let mut dirs = seq_dir
                    .path()
                    .read_dir()
                    .unwrap()
                    .collect::<Result<Vec<DirEntry>, _>>()
                    .unwrap();
                dirs.sort_by_key(|e| e.file_name().to_string_lossy().to_string());

                for thread_dir in dirs {
                    let dir_name = thread_dir.file_name().to_string_lossy().to_string();
                    if let Ok(thread_count) = u32::from_str(&dir_name) {
                        let csv_path = thread_dir.path().join("new").join("raw.csv");
                        let mut rdr = csv::ReaderBuilder::new()
                            .has_headers(true)
                            .from_path(csv_path)
                            .unwrap();

                        let mut byte_count = Option::<u64>::None;
                        let mut values: Vec<(Duration, u64)> = Vec::new();

                        for result in rdr.deserialize() {
                            let record: Record = result.unwrap();

                            if byte_count.is_none() {
                                byte_count = Some(record.throughput_num);
                            }

                            let duration =
                                build_duration(record.sample_measured_value, &record.unit);
                            let count = record.iteration_count;

                            if let Some((prev_dur, prev_count)) = values.last() {
                                values.push((duration - *prev_dur, count - *prev_count))
                            } else {
                                values.push((duration, count))
                            }
                        }

                        let mut point_estimates = values
                            .iter()
                            .map(|(d, c)| (*d) / (*c as u32))
                            .collect::<Vec<_>>();

                        point_estimates.sort_unstable();
                        let inliers = point_estimates[2..point_estimates.len() - 2].to_vec();

                        let mean_time = inliers.iter().sum::<Duration>() / inliers.len() as u32;
                        let mean_bps = byte_count.unwrap() as f64 / mean_time.as_secs_f64();

                        println!(
                            "| {thread_count:>10} | {mean_bps:>8.2e} b/s | {:>8.2?} | {:>11}/s |",
                            mean_time,
                            humansize::format_size_i(mean_bps, humansize::BINARY)
                        );
                    }
                }

                /*
                let thread_path = thread_dir.unwrap().path();
                let new_data_path = thread_path.join("new");

                let benchmark_path = new_data_path.join("benchmark.json");

                let value: serde_json::Value =
                    serde_json::from_reader(std::fs::File::open(benchmark_path).unwrap()).unwrap();

                println!("{:?}", value);
                 */
            }
        }
    }
}

fn build_duration(
    time: f64,
    units: &str,
) -> Duration {
    match units {
        "ns" => Duration::from_nanos(time as u64),
        "us" => Duration::from_micros(time as u64),
        "ms" => Duration::from_millis(time as u64),
        "s" => Duration::from_secs(time as u64),
        _ => panic!("unknown units: {}", units),
    }
}
