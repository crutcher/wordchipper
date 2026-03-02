use std::sync::Arc;

use wordchipper::{UnifiedTokenVocab, VocabIndex, vocab::io::write_base64_span_map};
use wordchipper_training::BPETRainerOptions;

use crate::{
    commands::lexers::LexerSelectorArgs,
    util::{input_batcher::InputBatcher, input_output::OutputArgs, logging::LogArgs},
};

/// File formats for the train command.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum FileFormat {
    /// Simple text files.
    Text,

    /// Parquet files.
    Parquet,
}

/// Args for the train command.
#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Input files.
    files: Vec<String>,

    /// Logging options.
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The input shard file format.
    #[arg(long, default_value = "text")]
    input_format: FileFormat,

    /// The input batch size.
    #[arg(long, default_value = "100")]
    input_batch_size: usize,

    /// Max vocab size.
    #[arg(long, default_value = "50281")]
    vocab_size: usize,

    #[command(flatten)]
    lexer_selector: LexerSelectorArgs,

    #[command(flatten)]
    output: OutputArgs,
}

impl TrainArgs {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;

        let pattern = self.lexer_selector.get_pattern()?;

        let mut trainer = BPETRainerOptions::new(pattern.clone(), self.vocab_size).init();

        log::info!("Reading shards:");
        InputBatcher::new(self.input_format, self.files.clone())
            .with_batch_size(self.input_batch_size)
            .for_each_batch(&mut |samples| {
                trainer.update_from_samples(samples.to_vec());
                Ok(true)
            })?;

        log::info!("Training Tokenizer...");
        let vocab: Arc<UnifiedTokenVocab<u32>> = trainer
            .train(Default::default())
            .expect("training failed")
            .into();

        log::info!("Vocabulary Size: {:?}", vocab.max_token().unwrap());

        if let Some(path) = &self.output.output {
            log::info!("output: {}", path);
        }
        let mut writer = self.output.open_writer()?;
        write_base64_span_map(vocab.span_vocab().span_map(), &mut writer)?;

        Ok(())
    }
}
