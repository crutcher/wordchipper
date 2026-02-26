use crate::commands::{cat::CatArgs, list_models::ListModelsArgs};

pub mod cat;
pub mod list_models;

/// Subcommands for wordchipper-cli
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Act as a streaming tokenizer.
    Cat(CatArgs),

    /// List available models.
    ListModels(ListModelsArgs),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Commands::Cat(cmd) => cmd.run(),
            Commands::ListModels(cmd) => cmd.run(),
        }
    }
}
