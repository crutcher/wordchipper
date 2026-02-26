pub mod cat_command;
pub mod common;
pub mod list_models_command;

use cat_command::CatArgs;
use clap::Parser;

use crate::list_models_command::ListModelsArgs;

/// wordchipper-cli
#[derive(clap::Parser, Debug)]
pub struct Args {
    /// Subcommand to run.
    #[clap(subcommand)]
    pub command: Commands,

    #[clap(flatten)]
    pub disk_cache: common::DiskCacheArgs,
}

/// Subcommands for wordchipper-cli
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Act as a streaming tokenizer.
    Cat(CatArgs),

    /// List available models.
    ListModels(ListModelsArgs),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match &args.command {
        Commands::Cat(cat_args) => cat_command::run_cat(&args, cat_args)?,
        Commands::ListModels(list_models_args) => {
            list_models_command::run_list_models(list_models_args)?
        }
    }

    Ok(())
}
