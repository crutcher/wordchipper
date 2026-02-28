mod cat;
mod models;
mod train;

/// Subcommands for wchipper
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Act as a streaming tokenizer.
    Cat(cat::CatArgs),

    /// Models sub-menu.
    Models(models::ModelsArgs),

    /// Train a new model.
    Train(train::TrainArgs),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Commands::Cat(cmd) => cmd.run(),
            Commands::Models(cmd) => cmd.run(),
            Commands::Train(cmd) => cmd.run(),
        }
    }
}
