use std::{
    collections::HashMap,
    path::Path,
};

use divan_parser::BenchResult;
use regex::Regex;
use wordchipper_cli_util::logging::LogArgs;

/// Args for the cat command.
#[derive(clap::Args, Debug)]
pub struct DevArgs {
    /// Path to the benchmark data.
    #[clap(long, default_value = "dev-crates/wordchipper-bench/bench-data")]
    data_dir: String,

    #[clap(flatten)]
    logging: LogArgs,
}

impl DevArgs {
    /// Run the dev command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;

        println!("{:?}", self);

        let data_dir = Path::new(&self.data_dir);

        let _data = load_parallel_data(data_dir.join("parallel"))?;

        Ok(())
    }
}

fn load_parallel_data<P: AsRef<Path>>(
    dir: P
) -> Result<HashMap<u32, Vec<BenchResult>>, Box<dyn std::error::Error>> {
    let dir = dir.as_ref();

    let pattern = r"^encoding_parallel\.(?<threads>\d+)\.json$";
    let re = Regex::new(pattern)?;

    let mut res: HashMap<u32, Vec<BenchResult>> = Default::default();

    for entry in dir.read_dir()?.filter_map(Result::ok) {
        let filename = entry.file_name().to_string_lossy().to_string();

        if let Some(caps) = re.captures(&filename) {
            let threads = caps.name("threads").unwrap().as_str().parse::<u32>()?;

            let file = std::fs::File::open(entry.path())?;
            let reader = std::io::BufReader::new(file);
            let data = serde_json::from_reader(reader)?;

            log::info!("threads: {}", threads);

            res.insert(threads, data);
        }
    }

    Ok(res)
}
