use std::{
    io::{BufRead, BufReader},
    sync::Arc,
};

use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    pretrained::openai::OA_R50K_BASE_PATTERN,
    vocab::io::write_base64_span_map,
};
use wordchipper_training::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};

use crate::{LogArgs, input_output::OutputArgs};

/// File formats for the train command.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum FileFormat {
    /// Simple text files.
    Text,
}

/// Args for the train command.
#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Input files.
    files: Vec<String>,

    #[clap(flatten)]
    pub logging: LogArgs,

    #[arg(long, default_value = "text")]
    input_format: FileFormat,

    /// Max vocab size.
    #[arg(long, default_value = "50281")]
    vocab_size: usize,

    /// Word span regex.
    #[arg(long, default_value_t = OA_R50K_BASE_PATTERN.as_str().to_string())]
    regex: String,

    #[command(flatten)]
    output: OutputArgs,
}

impl TrainArgs {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;

        let vocab_size = self.vocab_size;
        let options = BinaryPairVocabTrainerOptions::new(self.regex.clone(), vocab_size);

        let mut trainer: BinaryPairVocabTrainer<String, u32> = options.init();

        log::info!("Reading shards:");
        for (idx, path) in self.files.iter().enumerate() {
            log::info!("{idx}: {path}");
            match self.input_format {
                FileFormat::Text => {
                    self.read_text_file(&mut trainer, path)?;
                }
            }
        }

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

    fn read_text_file(
        &self,
        trainer: &mut BinaryPairVocabTrainer<String, u32>,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = BufReader::new(std::fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            trainer.update_from_samples(vec![line.to_string()]);
        }
        Ok(())
    }
}
