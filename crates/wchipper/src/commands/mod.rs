mod cat;
mod list_models;
mod train;

/// Subcommands for wchipper
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Act as a streaming tokenizer.
    Cat(cat::CatArgs),

    /// List available models.
    ListModels(list_models::ListModelsArgs),

    /// Train a new model.
    Train(train::TrainArgs),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Commands::Cat(cmd) => cmd.run(),
            Commands::ListModels(cmd) => cmd.run(),
            Commands::Train(cmd) => cmd.run(),
        }
    }
}
