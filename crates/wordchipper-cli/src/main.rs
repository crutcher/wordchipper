mod commands;
mod disk_cache;
mod input_output;
mod logging;
mod model_selector;
mod tokenizer_mode;

use clap::Parser;
use commands::Commands;

/// wordchipper-cli
#[derive(clap::Parser, Debug)]
pub struct Args {
    /// Subcommand to run.
    #[clap(subcommand)]
    pub command: Commands,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    args.command.run()
}
