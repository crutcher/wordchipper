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

/// Logging setup arg group.
///
/// This is in `main.rs` because `module_path!()` cares, and I haven't worked
/// out further questions of app logging to get around it.
///
/// This is a per-command mixin, because different commands
/// have different default levels they want to use.
#[derive(clap::Args, Debug)]
pub struct LogArgs {
    /// Silence log messages.
    #[clap(short, long)]
    pub quiet: bool,

    /// Turn debugging information on (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, default_value = None)]
    verbose: Option<u8>,

    /// Enable timestamped logging.
    #[clap(short, long)]
    pub ts: bool,
}

impl LogArgs {
    pub fn setup_logging(
        &self,
        default: u8,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let level = if let Some(verbose) = self.verbose
            && verbose > 0
        {
            verbose
        } else {
            default
        };

        let log_level = match level {
            0 => stderrlog::LogLevelNum::Off,
            1 => stderrlog::LogLevelNum::Error,
            2 => stderrlog::LogLevelNum::Warn,
            3 => stderrlog::LogLevelNum::Info,
            4 => stderrlog::LogLevelNum::Debug,
            _ => stderrlog::LogLevelNum::Trace,
        };

        stderrlog::new()
            .module(module_path!())
            .quiet(self.quiet)
            .verbosity(log_level)
            .init()?;

        Ok(())
    }
}
