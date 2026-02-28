use list_models::ListModelsArgs;

mod list_models;

/// Subcommands for the models command.
#[derive(clap::Subcommand, Debug)]
pub enum ModelsCommand {
    /// List available models.
    #[clap(visible_alias = "ls")]
    List(ListModelsArgs),
}

/// Args for the model listing command.
#[derive(clap::Args, Debug)]
pub struct ModelsArgs {
    #[clap(subcommand)]
    pub command: ModelsCommand,
}

impl ModelsArgs {
    /// Run the model listing command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.command {
            ModelsCommand::List(cmd) => cmd.run(),
        }
    }
}
