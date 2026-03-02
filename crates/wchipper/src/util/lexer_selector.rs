use crate::commands::lexers::LexerInventory;

#[derive(clap::Args, Debug)]
#[group(required = true, multiple = false)]
pub struct LexerSelectorArgs {
    /// Model name for selection.
    #[arg(long)]
    lexer_model: Option<String>,

    /// Pattern for selection.
    #[arg(long)]
    pattern: Option<String>,
}

impl LexerSelectorArgs {
    pub fn get_pattern(&self) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(p) = &self.pattern {
            return Ok(p.clone());
        }

        let name = self.lexer_model.as_ref().unwrap();
        match LexerInventory::build().find_model(name) {
            Some(model) => Ok(model.pattern.clone()),
            None => Err(format!("Model not found: {name}").into()),
        }
    }
}
